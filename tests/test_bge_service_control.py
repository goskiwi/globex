"""进程控制仅认可本项目身份，退出瞬态不误判为他人进程。"""
import json
import os
import pytest
from scripts import bge_service_control as control


@pytest.fixture
def process(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir()
    proc=tmp_path/'proc';proc.mkdir()
    folder=proc/'123';folder.mkdir()
    script=root/'service.py'
    monkeypatch.setattr(control,'ROOT',root)
    monkeypatch.setattr(control,'SCRIPT',script)
    monkeypatch.setattr(control,'PYTHON','/environment/python')
    monkeypatch.setattr(control,'PROC',proc)
    monkeypatch.setattr(control,'STATE',root/'process.json')
    (folder/'cmdline').write_bytes(b'/environment/python\0-B\0'+str(script).encode()+b'\0')
    (folder/'status').write_text(f'Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n')
    (folder/'stat').write_text('123 (python) S '+' '.join(['0']*18+['777']))
    (folder/'cwd').symlink_to(root,target_is_directory=True)
    return folder


def test_valid_process_and_pid_reuse(process):
    assert control.process_identity(123)=='777'
    control.STATE.write_text(json.dumps({'pid':123,'start_ticks':'wrong'}))
    with pytest.raises(RuntimeError,match='复用'):control.current()


def test_other_command_is_never_accepted(process):
    (process/'cmdline').write_bytes(b'/environment/python\0-B\0/other-person/job.py\0')
    with pytest.raises(RuntimeError,match='不属于'):control.process_identity(123)


def test_other_uid_is_never_accepted(process):
    (process/'status').write_text(f'Uid:\t{os.getuid()+1}\t0\t0\t0\n')
    with pytest.raises(RuntimeError,match='不属于'):control.process_identity(123)


def test_exiting_empty_command_and_missing_pid(process):
    (process/'cmdline').write_bytes(b'')
    assert control.process_identity(123) is None
    assert control.process_identity(999) is None


def test_zombie_is_stopped(process):
    (process/'stat').write_text('123 (python) Z '+' '.join(['0']*18+['777']))
    assert control.process_identity(123) is None
