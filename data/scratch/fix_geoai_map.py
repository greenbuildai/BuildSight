
import os

path = r'E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\src\components\GeoAIMap.tsx'

def fix_file():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. Fix the fragment at line 744 (index 743)
    # Original was likely {hoverCoords ? ( <> ... <> ) : ...}
    for i, line in enumerate(lines):
        if '743:' in str(i+1) or 'hoverCoords' in line:
            if '<>' in line and '?' in line:
                # We need to find the second <> and change it to </>
                # But let's be safer and just replace the whole block if we can find it
                pass

    # Actually, I have the line numbers now from my previous view_file
    # Line 742: <>
    # Line 744: <> (Bug)
    if '<>' in lines[741] and '<>' in lines[743]:
        lines[743] = lines[743].replace('<>', '</>')
        print("Fixed fragment at line 744")

    # 2. Inject the requested wrappers
    # We want them at line 751 (index 750)
    # After line 750: {/* Mode Controls - Premium Toggle */}
    # So we insert at 751
    opening_tags = [
        '        <div style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1000, display: "flex", flexDirection: "column", padding: "24px" }}> {/* Tactical HUD Layer */}\n',
        '          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", width: "100%" }}> {/* Top HUD Row */}\n'
    ]
    lines[751:751] = opening_tags
    print("Injected HUD Layer and Top HUD Row openings")

    # 3. Fix the end
    # The file ends at 914 (now 916 with our insertions)
    # We need to close:
    # 1. Top HUD Row
    # 2. Tactical HUD Layer
    # 3. Shell (Wrapper)
    # 4. Map Container (It is already closed if it's self-closing <div />)
    
    # Let's find the `style` tag end
    for i in range(len(lines)-1, 0, -1):
        if '}</style>' in lines[i]:
            # This is where we want to close our HUD layers before the final wrapper closing
            closing_tags = [
                '            </div> {/* Top HUD Row */}\n',
                '          </div> {/* Tactical HUD Layer */}\n'
            ]
            # Insert before the last </div>
            # Current structure:
            # ...style...
            # } </style>
            # </div> (Shell)
            # )
            # }
            lines.insert(i+1, closing_tags[0])
            lines.insert(i+2, closing_tags[1])
            print(f"Injected closing tags at index {i+1}")
            break

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("File saved successfully.")

if __name__ == "__main__":
    fix_file()
