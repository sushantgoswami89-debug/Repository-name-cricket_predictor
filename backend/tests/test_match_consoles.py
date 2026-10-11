import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_live_auto import _MatchConsoles


def test_each_match_prints_into_its_own_file_and_driver_lines_stay_in_the_window(tmp_path):
    window = io.StringIO()
    consoles = _MatchConsoles(window, tmp_path)
    consoles.write("driver line\n")
    with consoles.match("151587"):
        consoles.write("[IND-WI] 7.1  1 run\n")
    with consoles.match("170690"):
        consoles.write("[PAK-SL] 13.1  FOUR\n")
    consoles.write("scan done\n")
    assert window.getvalue() == "driver line\nscan done\n"
    assert "[IND-WI] 7.1  1 run" in consoles.path("151587").read_text()
    assert "PAK-SL" not in consoles.path("151587").read_text()
    assert "[PAK-SL] 13.1  FOUR" in consoles.path("170690").read_text()
