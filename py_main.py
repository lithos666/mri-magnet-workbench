r"""MRI磁体设计入口。所有日常设计参数和约束请修改同目录 parameters.py。

用法：
  .venv\Scripts\python.exe py_main.py check
  .venv\Scripts\python.exe py_main.py all --quick
"""
import sys

def main():
    if len(sys.argv)>1 and sys.argv[1]=='workbench':
        from mri_magnet.workbench_cli import main as workbench_main
        return workbench_main(sys.argv[2:])
    from mri_magnet.cli import main as legacy_main
    return legacy_main()

if __name__ == '__main__':
    raise SystemExit(main())
