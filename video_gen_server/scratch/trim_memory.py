import ctypes
import sys

pid = 38620
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

# PROCESS_ALL_ACCESS = 0x1F0FFF
handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
if handle:
    res = psapi.EmptyWorkingSet(handle)
    kernel32.CloseHandle(handle)
    print("EmptyWorkingSet returned:", res)
else:
    print("Failed to open process", kernel32.GetLastError())
