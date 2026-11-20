from mas.benchmarks.tasks import generate_synthetic, check, split, Task, load_gsm8k_jsonl

def test_generate_and_check():
    ts = generate_synthetic(20, seed=1)
    assert len(ts) == 20 and {t.difficulty for t in ts} == {1, 2, 3, 4}
    assert all(check(t, t.answer) for t in ts)
    assert not check(ts[0], None) and not check(ts[0], "abc")

def test_split_disjoint():
    ts = generate_synthetic(40)
    a, b = split(ts)
    assert not {t.id for t in a} & {t.id for t in b}

def test_gsm_loader(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"question":"1 and 2 and 3 and 4","answer":"x #### 1,234"}\n')
    t = load_gsm8k_jsonl(p)
    assert t[0].answer == "1234" and check(t[0], "1234")
