import sys

sys.path.append("../")
from build.pybind_interface.ReID import ReID
import cv2
import time


if __name__ == '__main__':
    iter_ = 20000
    m = ReID(0)
    m.build("../build/kd_r18_distill.engine")
    print("build done")
    
    frame = cv2.imread("/data/sunp/algorithm/2020_1015_time/pytorchtotensorrt_reid/test/query/0001/0001_c1s1_001051_00.jpg")
    m.infer(frame)
    t0 = time.time()

    for i in range(iter_):
        m.infer(frame)

    total = time.time() - t0
    print("CPP API fps is {:.1f}, avg infer time is {:.2f}ms".format(iter_ / total, total / iter_ * 1000))