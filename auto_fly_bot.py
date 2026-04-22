import pydirectinput
import time
import random
import re
import numpy as np
from mcrcon import MCRcon

# ================= 配置参数 =================
COLLECT_DURATION_MINUTES = 60
RCON_IP = "127.0.0.1"
RCON_PASS = "6567"
PLAYER_NAME = "Nickyaso"
STUCK_THRESHOLD_VELOCITY = 0.5
STUCK_TIME_LIMIT = 3.0


# ============================================

def smart_auto_fly():
    print("=== 智能状态感知采集 Bot 4.0 ===")
    time.sleep(5)

    start_time = time.time()
    end_time = start_time + (COLLECT_DURATION_MINUTES * 60)

    last_x, last_z = None, None
    stuck_start_time = None
    stats = {"moves": 0, "fireworks": 0, "recoveries": 0}

    try:
        with MCRcon(RCON_IP, RCON_PASS) as mcr:
            pydirectinput.keyDown('w')

            while time.time() < end_time:
                raw_pos = mcr.command(f"data get entity {PLAYER_NAME} Pos")
                match = re.search(r'\[(.*?)d, (.*?)d, (.*?)d\]', raw_pos)

                if match:
                    x, z = float(match.group(1)), float(match.group(3))
                    if last_x is not None:
                        v_xz = np.sqrt((x - last_x) ** 2 + (z - last_z) ** 2)

                        if v_xz < STUCK_THRESHOLD_VELOCITY:
                            if stuck_start_time is None:
                                stuck_start_time = time.time()
                            elif time.time() - stuck_start_time > STUCK_TIME_LIMIT:
                                print(f"\n[AI判定] 检测到真实卡死 (V={v_xz:.2f})，启动逃逸...")
                                pydirectinput.keyUp('w')
                                for _ in range(5): pydirectinput.moveRel(0, -120); time.sleep(0.01)
                                pydirectinput.press('space');
                                time.sleep(0.1)
                                pydirectinput.press('space');
                                pydirectinput.click(button='right')
                                time.sleep(2.0)
                                pydirectinput.keyDown('w')
                                stuck_start_time = None
                                stats["recoveries"] += 1
                        else:
                            stuck_start_time = None

                    last_x, last_z = x, z

                if stuck_start_time is None:
                    move_x = random.randint(-60, 60)
                    move_y = random.randint(-10, 5)
                    for _ in range(3):
                        pydirectinput.moveRel(int(move_x / 3), int(move_y / 3))
                        time.sleep(0.02)
                    if random.random() < 0.1:
                        pydirectinput.click(button='right')
                        stats["fireworks"] += 1

                time.sleep(0.5)
                stats["moves"] += 1

    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        pydirectinput.keyUp('w')
        print(f"任务结束。总计执行有效逃逸: {stats['recoveries']} 次。")


if __name__ == "__main__":
    smart_auto_fly()