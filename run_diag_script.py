from test_worker_diag import run_diag

cmd = r"""python3 -c "
with open('/handler.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = [
    '    except Exception as e:\n',
    '        comfy_log_tail = \"\"\n',
    '        try:\n',
    '            with open(\"/ComfyUI/comfyui.log\", \"r\", errors=\"ignore\") as lf:\n',
    '                comfy_log_tail = \"\".join(lf.readlines()[-60:])\n',
    '        except Exception:\n',
    '            pass\n',
    '        logger.error(f\"❌ ComfyUI 오류/크래시: {e}\\nComfyUI log:\\n{comfy_log_tail}\")\n',
    '        return {\"error\": f\"{type(e).__name__}: {str(e)}\", \"comfyui_log\": comfy_log_tail}\n',
]

for i, line in enumerate(lines):
    if line.strip() == 'finally:':
        # Check if preceding line is in get_videos try block
        if any('get_videos' in lines[k] for k in range(max(0, i-10), i)):
            lines = lines[:i] + new_lines + lines[i:]
            with open('/handler.py', 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print('Inserted successfully before finally at line', i)
            break
else:
    print('finally not found')
"
"""
run_diag(cmd)
