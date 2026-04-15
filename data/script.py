import struct
import time
import csv
import os
import logging
from datetime import datetime

try:
    import snap7
    from snap7.util import get_bool, get_real
except ImportError:
    raise ImportError(
        "python-snap7 is required.  Install it with:  pip install python-snap7 snap7"
    )

# ── Configuration ──────────────────────────────────────────────────────────────

PLC_IP   = "192.168.0.1"   # ← change to your PLC IP
RACK     = 0               # S7-1200 / 1500 → rack 0
SLOT     = 1               # S7-1200 → slot 1 | S7-1500 → slot 1

POLL_INTERVAL_S = 0.5      # seconds between reads while running

OUTPUT_DIR  = "."          # folder where CSV files are saved
CSV_PREFIX  = "pid_tank_level"

# ── Tag / Address Map ──────────────────────────────────────────────────────────
# Each entry: (area, db_number, start_byte, data_type, column_name)
# area codes: 0x83 = M (Merkers), 0x84 = DB
# For REAL (float) use start bytes that are DWORD-aligned (multiple of 4).

M_AREA = snap7.type.Areas.MK    # Merker / flag memory

TAG_MAP = [
    # (area,   db, byte,  type,   csv_column)
    (M_AREA,   0,  100,  "REAL",  "Setpoint_%"),
    (M_AREA,   0,  104,  "REAL",  "Level_PV_%"),
    (M_AREA,   0,  108,  "REAL",  "PID_Output_%"),
    (M_AREA,   0,  112,  "REAL",  "Error"),
    (M_AREA,   0,  116,  "REAL",  "P_Term"),
    (M_AREA,   0,  120,  "REAL",  "I_Term"),
    (M_AREA,   0,  124,  "REAL",  "D_Term"),
]

# M20.0 – Start button
START_BIT_BYTE = 20   # byte 20  (M20.x)
START_BIT_BIT  = 0    # bit  0   (Mx.0)

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PID_Logger")


# ── Helpers ────────────────────────────────────────────────────────────────────

def bytes_to_real(raw: bytes) -> float:
    """Convert 4-byte big-endian S7 REAL to Python float."""
    return struct.unpack(">f", raw)[0]


def read_bool_m(client: snap7.client.Client, byte: int, bit: int) -> bool:
    """Read a single BOOL from M-area."""
    raw = client.read_area(M_AREA, 0, byte, 1)
    return get_bool(raw, 0, bit)


def read_real_m(client: snap7.client.Client, byte: int) -> float:
    """Read a REAL (4 bytes) from M-area."""
    raw = client.read_area(M_AREA, 0, byte, 4)
    return bytes_to_real(raw)


def read_all_tags(client: snap7.client.Client) -> dict:
    """Read every tag in TAG_MAP and return {column: value}."""
    row = {}
    for area, db, byte, dtype, col in TAG_MAP:
        try:
            if dtype == "REAL":
                raw = client.read_area(area, db, byte, 4)
                row[col] = round(bytes_to_real(raw), 4)
            # extend here for INT, DINT, BOOL, etc. if needed
        except Exception as exc:
            log.warning("Could not read %s @ byte %d: %s", col, byte, exc)
            row[col] = None
    return row


def new_csv_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{CSV_PREFIX}_{ts}.csv")


# ── Main logger loop ───────────────────────────────────────────────────────────

def run():
    client = snap7.client.Client()

    log.info("Connecting to PLC at %s (rack=%d, slot=%d) …", PLC_IP, RACK, SLOT)
    try:
        client.connect(PLC_IP, RACK, SLOT)
    except Exception as exc:
        log.error("Connection failed: %s", exc)
        return

    if not client.get_connected():
        log.error("PLC not connected – check IP / rack / slot.")
        return

    log.info("Connected.  Waiting for M20.0 (Start button) to go HIGH …")

    csv_file   = None
    csv_writer = None
    recording  = False

    csv_columns = ["Timestamp"] + [col for *_, col in TAG_MAP]

    try:
        while True:
            # ── read start bit ──
            try:
                start_active = read_bool_m(client, START_BIT_BYTE, START_BIT_BIT)
            except Exception as exc:
                log.error("Failed to read M20.0: %s", exc)
                time.sleep(2)
                continue

            # ── rising edge → start recording ──
            if start_active and not recording:
                csv_path = new_csv_path()
                log.info("START detected → recording to %s", csv_path)

                csv_file   = open(csv_path, "w", newline="")
                csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                csv_writer.writeheader()
                recording  = True

            # ── falling edge → stop recording ──
            elif not start_active and recording:
                log.info("STOP detected → closing CSV file.")
                csv_file.close()
                csv_file   = None
                csv_writer = None
                recording  = False
                log.info("Waiting for M20.0 to go HIGH again …")

            # ── collect data while active ──
            if recording:
                row = read_all_tags(client)
                row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                csv_writer.writerow(row)
                csv_file.flush()   # write to disk immediately

                log.debug(
                    "SP=%.2f%%  PV=%.2f%%  OUT=%.2f%%",
                    row.get("Setpoint_%", 0),
                    row.get("Level_PV_%", 0),
                    row.get("PID_Output_%", 0),
                )

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – shutting down.")

    finally:
        if csv_file and not csv_file.closed:
            csv_file.close()
            log.info("CSV file closed.")
        client.disconnect()
        log.info("PLC disconnected.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
```