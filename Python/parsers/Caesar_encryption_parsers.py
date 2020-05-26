from collections import deque
from string import ascii_lowercase, ascii_uppercase
import argparse

def caesar_cipher_encryption(plaintext, key):
    upper = deque(ascii_uppercase)
    upper.rotate(key)
    upper_str = ''.join(list(upper))

    lower = deque(ascii_lowercase)
    lower.rotate(key)
    lower_str = ''.join(list(lower))

    return plaintext.translate(str.maketrans(ascii_uppercase, upper_str)).translate(str.maketrans(ascii_lowercase, lower_str))


def caesar_cipher_decryption(plaintext, key):
    upper = deque(ascii_uppercase)
    upper.rotate(-key)
    upper_str = ''.join(list(upper))

    lower = deque(ascii_lowercase)
    lower.rotate(-key)
    lower_str = ''.join(list(lower))

    return plaintext.translate(str.maketrans(ascii_uppercase, upper_str)).translate(str.maketrans(ascii_lowercase, lower_str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--plaintext", help="Plain text", type=str, required=True)
    parser.add_argument("-K", "--key", help="key", type=int, required=True)

    args = parser.parse_args()
    plaintext = args.plaintext
    key = args.key

    ciphertext = caesar_cipher_encryption(plaintext, key)
    retreived_plaintext = caesar_cipher_decryption(ciphertext, key)

    print(f"Plaintext = {plaintext}")
    print(f"Key = {key}")
    print(f"Ciphertext = {ciphertext}")
    print(f"Retreived plaintext = {retreived_plaintext}")

if __name__ == '__main__':
    main()