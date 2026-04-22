import re
import csv
import time
from mcrcon import MCRcon

# ================= 配置参数 =================
RCON_IP = "127.0.0.1"
RCON_PASS = "6567"
PLAYER_NAME = "Nickyaso"
DURATION_MINUTES = 60
HZ = 10


# ============================================

def collect():
    interval = 1.0 / HZ
    try:
        with MCRcon(RCON_IP, RCON_PASS) as mcr:
            print("连接成功！开始记录 5D 基础特征 (X, Y, Z, Yaw, Pitch)...")
            mcr.command("gamerule sendCommandFeedback false")

            with open('training_data.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'x', 'y', 'z', 'yaw', 'pitch'])

                start_time = time.time()
                count = 0

                while time.time() - start_time < DURATION_MINUTES * 60:
                    loop_start = time.perf_counter()

                    raw_pos = mcr.command(f"data get entity {PLAYER_NAME} Pos")
                    match_pos = re.search(r'\[(.*?)d, (.*?)d, (.*?)d\]', raw_pos)
                    raw_rot = mcr.command(f"data get entity {PLAYER_NAME} Rotation")
                    match_rot = re.search(r'\[(.*?)f, (.*?)f\]', raw_rot)

                    if match_pos and match_rot:
                        x, y, z = [float(match_pos.group(i)) for i in range(1, 4)]
                        yaw, pitch = float(match_rot.group(1)), float(match_rot.group(2))

                        writer.writerow([time.time(), x, y, z, yaw, pitch])
                        count += 1
                        if count % 100 == 0:
                            print(f"已记录 {count} 条")

                    elapsed = time.perf_counter() - loop_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    collect()