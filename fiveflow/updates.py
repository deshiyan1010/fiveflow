"""Repository updates shared by startup and the shell command."""
from pathlib import Path
import subprocess
import sys


def _git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                          text=True, check=True, timeout=15).stdout.strip()


def available_update(root):
    root = Path(root)
    if not (root / '.git').exists():
        return None
    # Respect the checkout's upstream; installer clones default to origin/master.
    remotes = _git(root, 'remote').splitlines()
    try:
        target = _git(root, 'rev-parse', '--abbrev-ref', '@{upstream}')
        remote = _git(root, 'config', '--get', f'branch.{_git(root, "branch", "--show-current")}.remote')
    except subprocess.CalledProcessError:
        remote = 'origin' if 'origin' in remotes else (remotes[0] if len(remotes) == 1 else None)
        if remote is None:
            return None
        target = f'{remote}/master'
    if remote == '.':
        return None
    _git(root, 'fetch', remote, '--quiet')
    behind = _git(root, 'rev-list', '--count', f'HEAD..{target}')
    return target if int(behind) else None


def apply_update(root, target):
    root = Path(root)
    # Reject divergence before changing files, and preserve all user edits.
    if _git(root, 'status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('Local changes exist; commit or stash them before updating')
    _git(root, 'merge-base', '--is-ancestor', 'HEAD', target)
    previous = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'merge', '--ff-only', target)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                        str(root / 'requirements.txt')], check=True)
    except (OSError, subprocess.CalledProcessError):
        # Keep code compatible with the environment if dependency installation
        # fails. Only undo our fast-forward if nobody edited the tree meantime.
        if not _git(root, 'status', '--porcelain', '--untracked-files=normal'):
            _git(root, 'reset', '--keep', previous)
        raise


def main():
    root = Path(__file__).resolve().parents[1]
    try:
        target = available_update(root)
        if target:
            apply_update(root, target)
            print('[fiveflow] Updated. Restart fiveflow to use the new version.')
        else:
            # Also repair missing dependencies when already at the latest code.
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                            str(root / 'requirements.txt')], check=True)
            print('[fiveflow] Already up to date.')
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[fiveflow] Update failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
