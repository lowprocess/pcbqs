"""_cli.load_board: KiCad imports cached by content hash and net subset."""
import pytest

from pcbqs import _cli

kicad2ems = pytest.importorskip('kicad2ems.board')


class FakeBoard:
    calls = 0

    def __init__(self, raw, **kw):
        FakeBoard.calls += 1
        self.raw, self.kw = raw, kw

    def to_dict(self):
        return dict(source=self.raw, nets=self.kw.get('nets'))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    import kicad2ems.importers
    FakeBoard.calls = 0
    monkeypatch.setattr(kicad2ems.board, 'Board', FakeBoard)
    monkeypatch.setattr(kicad2ems.importers, 'open_board', lambda path: open(path).read())
    monkeypatch.setenv('PCBQS_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.delenv('PCBQS_NO_CACHE', raising=False)
    pcb = tmp_path / 'b.kicad_pcb'
    pcb.write_text('(kicad_pcb v1)')
    return pcb


def test_second_load_comes_from_the_cache(fake):
    a = _cli.load_board(fake)
    b = _cli.load_board(fake)
    assert a == b and FakeBoard.calls == 1
    assert len(list(_cli.cache_dir().glob('*.pickle'))) == 1


def test_an_edited_board_or_another_net_subset_is_imported_again(fake):
    _cli.load_board(fake)
    _cli.load_board(fake, {'GND', 'VIN'})
    assert FakeBoard.calls == 2
    fake.write_text('(kicad_pcb v2)')
    assert _cli.load_board(fake)['source'] == '(kicad_pcb v2)'
    assert FakeBoard.calls == 3
    _cli.load_board(fake, {'VIN', 'GND'})                    # the subset's order does not matter
    assert FakeBoard.calls == 4                              # (v2 with these nets: new)
    _cli.load_board(fake, {'GND', 'VIN'})
    assert FakeBoard.calls == 4


def test_cache_can_be_bypassed(fake, monkeypatch):
    monkeypatch.setenv('PCBQS_NO_CACHE', '1')
    _cli.load_board(fake)
    _cli.load_board(fake)
    assert FakeBoard.calls == 2
    assert not _cli.cache_dir().exists()


def test_a_damaged_entry_is_replaced(fake):
    _cli.load_board(fake)
    entry, = _cli.cache_dir().glob('*.pickle')
    entry.write_bytes(b'not a pickle')
    assert _cli.load_board(fake)['source'] == '(kicad_pcb v1)'
    assert FakeBoard.calls == 2
    assert _cli.load_board(fake) and FakeBoard.calls == 2
