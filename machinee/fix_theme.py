import os

file_path = 'app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Plotly white theme with dark theme
content = content.replace('template="plotly_white"', 'template="plotly_dark"')
content = content.replace('font=dict(color="#000000")', 'font=dict(color="#f8fafc")')

# Replace hardcoded light HTML styles in the simulator diagnosis block
content = content.replace('background: #f8fafc; padding: 15px; border-radius: 12px; border: 1px solid rgba(15,23,42,0.08);', 'background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);')
content = content.replace('color: #475569;', 'color: #94a3b8;')

# Fix the c-dark class usage in headers for dark mode
# We want c-dark to be white or light blue in dark mode, or better just inherit the #f8fafc color
# But instead of changing app.py, we can change icons.html .c-dark to have #f8fafc

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated app.py successfully!")
