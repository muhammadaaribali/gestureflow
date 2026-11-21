"""Control media and system audio from a Linux desktop session."""
import shutil
import subprocess


def make_system_backend(volume_step=3):
    if not isinstance(volume_step, int) or not 1 <= volume_step <= 100:
        raise ValueError("volume_step_percent must be an integer from 1 to 100")

    step = f"{volume_step}%"
    commands = {}
    if shutil.which("playerctl"):
        commands.update(play_pause=["playerctl", "play-pause"],
                        next_track=["playerctl", "next"],
                        prev_track=["playerctl", "previous"])
    if shutil.which("wpctl"):
        sink = "@DEFAULT_AUDIO_SINK@"
        commands.update(volume_up=["wpctl", "set-volume", "-l", "1.0", sink, step + "+"],
                        volume_down=["wpctl", "set-volume", sink, step + "-"],
                        mute=["wpctl", "set-mute", sink, "toggle"])
    elif shutil.which("pactl"):
        sink = "@DEFAULT_SINK@"
        commands.update(volume_up=["pactl", "set-sink-volume", sink, "+" + step],
                        volume_down=["pactl", "set-sink-volume", sink, "-" + step],
                        mute=["pactl", "set-sink-mute", sink, "toggle"])
    elif shutil.which("amixer"):
        master = ["amixer", "-q", "set", "Master"]
        commands.update(volume_up=master + [step + "+"],
                        volume_down=master + [step + "-"], mute=master + ["toggle"])

    last_player = None

    def backend(action):
        nonlocal last_player
        command = commands.get(action)
        if command is None:
            print(f"[NO TOOL] {action}", flush=True)
            return False
        try:
            if command[0] == "playerctl":
                result = subprocess.run(
                    ["playerctl", "-a", "status", "-f", "{{playerInstance}} {{status}}"],
                    check=True, capture_output=True, text=True, timeout=1,
                )
                players = dict(line.rsplit(" ", 1) for line in result.stdout.splitlines() if " " in line)
                if not players:
                    print("[NO PLAYER] Start a video or song first.", flush=True)
                    return False
                # Follow playback. Keep the same player when a gesture pauses it.
                def priority(player):
                    return (players[player] == "Playing", players[player] == "Paused",
                            player == last_player)

                target = max(players, key=priority)
                command = ["playerctl", "-p", target, command[1]]
            subprocess.run(command, check=True, timeout=1, capture_output=True)
            if command[0] == "playerctl":
                last_player = target
        except (OSError, subprocess.SubprocessError) as error:
            print(f"[FAILED] {action}: {error}", flush=True)
            return False
        destination = f" ({last_player})" if command[0] == "playerctl" else ""
        print(f"[OK] {action}{destination}", flush=True)
        return True

    return backend, {action: cmd[0] for action, cmd in commands.items()}
