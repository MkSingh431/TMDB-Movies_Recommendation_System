import runpy
import traceback

try:
    runpy.run_path('app.py', run_name='__main__')
except Exception:
    traceback.print_exc()