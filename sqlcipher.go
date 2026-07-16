// Package sqlcipher implements the SQLCipher 4 on-disk page format in pure Go.
//
// It is a port of the SQLCipher codec (https://github.com/sqlcipher/sqlcipher,
// v4.5.6, BSD-3, Copyright (c) ZETETIC LLC) — see NOTICE. It knows nothing about
// database/sql or SQLite drivers: it is the format, and only the format. Anything
// that must read or write SQLCipher files — a driver, a backup or replication
// path, a migration or a forensic tool — uses this package directly.
//
// # The format
//
// A SQLCipher database is a SQLite database whose pages are individually
// encrypted, with per-page reserve bytes carrying the IV and authentication tag:
//
//	page N on disk:  [ ciphertext | IV (16) | HMAC-SHA512 (64) ]
//	page 1 on disk:  [ salt (16) | ciphertext | IV (16) | HMAC-SHA512 (64) ]
//
// Pages are independent: no cross-page state, no chaining. The salt is the first
// SaltSize bytes of page 1 and is stored in the clear — it is the only part of
// the file that is not ciphertext. Because page 1's first SaltSize bytes hold the
// salt instead of SQLite's "SQLite format 3\x00" magic, page 1 encrypts only the
// bytes after that offset; Decrypt restores the magic, which is what SQLite's
// pager expects to see.
//
// The ciphertext is AES-256-CBC with no padding, so its length is always a
// multiple of the AES block size. The IV is fresh on every page write. The HMAC
// covers ciphertext || IV || page number (little-endian uint32), which
// authenticates the page contents, the IV, and the page's position in the file —
// so pages cannot be reordered, and a modified IV is detected.
//
// # Keying
//
// A Key is either a raw 32-byte key (no KDF — SQLCipher's x'HEX' form) or a
// passphrase (PBKDF2-HMAC-SHA512, 256000 iterations by default). Either way the
// page-authentication key is a second, distinct key derived from the page
// encryption key using the salt masked with 0x3a and 2 PBKDF2 iterations.
//
// # Failing closed
//
// Decrypt authenticates before it decrypts and returns ErrKey on any HMAC
// mismatch. A wrong key errors; it never returns garbage plaintext.
package sqlcipher

import (
	"bytes"
	"crypto/hmac"
	"crypto/pbkdf2"
	"crypto/sha512"
	"errors"
	"fmt"
	"io"
)

// Format sizes, all fixed by SQLCipher 4.
const (
	SaltSize = 16 // KDF salt: the first bytes of page 1, stored in the clear
	KeySize  = 32 // AES-256
	IVSize   = 16 // AES block size
	HMACSize = 64 // SHA-512 digest

	// Reserve is the per-page trailer SQLite must leave free at the end of every
	// page: IV || HMAC, rounded up to a multiple of the AES block size (it is
	// already a multiple, so 16+64=80). SQLite records it in byte 20 of the
	// database header, so a database carries its own reserve size.
	Reserve = IVSize + HMACSize

	// DefaultPageSize and DefaultIter are SQLCipher 4's defaults.
	DefaultPageSize = 4096
	DefaultIter     = 256000

	blockSize = 16   // AES
	fastIter  = 2    // PBKDF2 iterations for the page-authentication key
	saltMask  = 0x3a // masks the salt so the two keys derive from distinct salts
)

// magic is the SQLite file header. It occupies page 1's first SaltSize bytes in
// a plaintext database, where an encrypted database keeps the salt instead.
var magic = []byte("SQLite format 3\x00")

// ErrKey reports that a page did not authenticate: the key is wrong, or the page
// was corrupted or tampered with. The two are deliberately indistinguishable.
var ErrKey = errors.New("sqlcipher: wrong key or corrupted page")

// Params are the format parameters that vary between databases. The zero value
// means SQLCipher 4 defaults; set a field only to interoperate with a database
// that was written with a non-default PRAGMA.
type Params struct {
	PageSize int // cipher_page_size; 0 means DefaultPageSize
	Iter     int // kdf_iter, passphrase keying only; 0 means DefaultIter
}

func (p Params) pageSize() int {
	if p.PageSize == 0 {
		return DefaultPageSize
	}
	return p.PageSize
}

func (p Params) iter() int {
	if p.Iter == 0 {
		return DefaultIter
	}
	return p.Iter
}

// Key is database key material: a raw key or a passphrase. It is inert until
// bound to a database salt by NewCodec.
type Key struct {
	raw  []byte
	pass string
}

// RawKey uses key directly as the page encryption key, skipping the passphrase
// KDF. key must be KeySize bytes. This is SQLCipher's x'HEX' keying form and the
// one Hanzo uses: keys come from KMS already uniformly random, so a KDF over them
// would buy nothing.
func RawKey(key []byte) Key { return Key{raw: key} }

// Passphrase derives the page encryption key from pass via PBKDF2-HMAC-SHA512.
func Passphrase(pass string) Key { return Key{pass: pass} }

// Codec encrypts and decrypts the pages of one database. It is immutable and safe
// for concurrent use.
type Codec struct {
	pageSize int
	salt     []byte // SaltSize bytes, page 1's plaintext prefix
	key      []byte // page encryption key
	mac      []byte // page authentication key
	rand     io.Reader
}

// NewCodec binds key material to a database salt. salt is SaltSize bytes, read
// from the first bytes of page 1 of an existing database (see FileSalt), or freshly
// random for a new one.
//
// Deriving from a passphrase runs Params.Iter PBKDF2 iterations and is
// deliberately slow; do it once per database, not once per page.
func NewCodec(k Key, salt []byte, p Params) (*Codec, error) {
	if len(salt) != SaltSize {
		return nil, fmt.Errorf("sqlcipher: salt must be %d bytes, got %d", SaltSize, len(salt))
	}
	if n := p.pageSize(); n < 512 || n > 65536 || n&(n-1) != 0 {
		return nil, fmt.Errorf("sqlcipher: page size %d is not a power of two in [512,65536]", n)
	}

	var key []byte
	switch {
	case k.raw != nil:
		if len(k.raw) != KeySize {
			return nil, fmt.Errorf("sqlcipher: raw key must be %d bytes, got %d", KeySize, len(k.raw))
		}
		key = k.raw
	case k.pass != "":
		var err error
		if key, err = pbkdf2.Key(sha512.New, k.pass, salt, p.iter(), KeySize); err != nil {
			return nil, fmt.Errorf("sqlcipher: derive key: %w", err)
		}
	default:
		return nil, errors.New("sqlcipher: no key material")
	}

	// The page-authentication key derives from the page encryption key over the
	// masked salt, so the two keys are distinct but both reproducible from the
	// passphrase and the file.
	masked := make([]byte, SaltSize)
	for i, b := range salt {
		masked[i] = b ^ saltMask
	}
	mac, err := pbkdf2.Key(sha512.New, string(key), masked, fastIter, KeySize)
	if err != nil {
		return nil, fmt.Errorf("sqlcipher: derive page-authentication key: %w", err)
	}

	return &Codec{pageSize: p.pageSize(), salt: bytes.Clone(salt), key: bytes.Clone(key), mac: mac}, nil
}

// PageSize is the on-disk size of every page, including the reserve trailer.
func (c *Codec) PageSize() int { return c.pageSize }

// Salt returns the database salt.
func (c *Codec) Salt() []byte { return bytes.Clone(c.salt) }

// FileSalt reads the salt from page 1 of an encrypted database. page1 need only
// be the first SaltSize bytes of the file.
func FileSalt(page1 []byte) ([]byte, error) {
	if len(page1) < SaltSize {
		return nil, fmt.Errorf("sqlcipher: need %d bytes to read the salt, got %d", SaltSize, len(page1))
	}
	return bytes.Clone(page1[:SaltSize]), nil
}

// offset is the number of leading bytes page pgno does not encrypt: page 1 keeps
// the salt in the clear, every other page encrypts from byte 0.
func offset(pgno uint32) int {
	if pgno == 1 {
		return SaltSize
	}
	return 0
}

// pageMAC authenticates ciphertext||IV for page pgno. Binding the page number
// stops a valid page from being replayed at another position in the file; it is
// little-endian to match SQLCipher's default (CIPHER_FLAG_LE_PGNO), which is
// a fixed byte order rather than the host's so files stay portable.
func (c *Codec) pageMAC(pgno uint32, data []byte) []byte {
	m := hmac.New(sha512.New, c.mac)
	m.Write(data)
	m.Write([]byte{byte(pgno), byte(pgno >> 8), byte(pgno >> 16), byte(pgno >> 24)})
	return m.Sum(nil)
}
