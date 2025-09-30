
from tools.schedules import parse_linear, parse_piecewise

def main():
    f = parse_linear("linear:100->200@0.5", total_steps=100)
    assert f(0) == 100 and f(50) == 200 and f(100) == 200
    g = parse_piecewise("step:8->4@0.5;4->2@0.8", total_steps=100)
    assert g(10) == 4 and g(60) == 2 and g(90) == 2
    print("OK: schedules parsing")

if __name__ == "__main__":
    main()
