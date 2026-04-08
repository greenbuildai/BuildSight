#!/usr/bin/env python
"""Fix double-encoded UTF-8 (mojibake) in comparative_study.md"""

path = 'e:/Company/Green Build AI/Prototypes/BuildSight/docs/comparative_study.md'
with open(path, 'rb') as f:
    raw = f.read()

text = raw.decode('utf-8')

# Try ftfy first (best approach)
try:
    import ftfy
    fixed = ftfy.fix_text(text)
    method = 'ftfy'
except ImportError:
    # Manual byte-level fix: re-encode as latin-1 where possible to reverse double-encoding
    # For chars that can't encode to latin-1, keep as-is
    fixed_chars = []
    i = 0
    while i < len(text):
        c = text[i]
        # Check if this starts a mojibake sequence by trying to re-encode as latin-1
        try:
            b = c.encode('latin-1')
            # If it's a high latin-1 char (>127), it may be part of mojibake
            if b[0] > 127 and i + 1 < len(text):
                # Try to collect bytes for a multi-byte UTF-8 sequence
                seq = c
                j = i + 1
                while j < len(text) and j < i + 6:
                    try:
                        seq += text[j]
                        seq_bytes = seq.encode('latin-1')
                        try:
                            recovered = seq_bytes.decode('utf-8')
                            # Success - use the recovered char if it's printable
                            if len(recovered) == 1 and ord(recovered) > 127:
                                fixed_chars.append(recovered)
                                i = j + 1
                                break
                        except (UnicodeDecodeError, ValueError):
                            pass
                        j += 1
                    except (UnicodeEncodeError, ValueError):
                        break
                else:
                    fixed_chars.append(c)
                    i += 1
            else:
                fixed_chars.append(c)
                i += 1
        except (UnicodeEncodeError, ValueError):
            fixed_chars.append(c)
            i += 1
    fixed = ''.join(fixed_chars)
    method = 'manual'

with open(path, 'w', encoding='utf-8') as f:
    f.write(fixed)

lines = fixed.splitlines()
print("Method:", method)
print("Line 3:", lines[2] if len(lines) > 2 else '')
print("Line 4:", lines[3] if len(lines) > 3 else '')
suspect = fixed.count('\xc3') + fixed.count('\xe2\x80')
print("Done. Total lines:", len(lines))
