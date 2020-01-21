"""
RANSAC algorithm pseudo code
"""


def ransacMatching(A, B):
    ininliers = random pick 4 pairs of points from A and B
    model = pick a model
    run_time = times to run

    Loop(0, run_time):
        H = calculate the homegraphy with new ininliers based on the model
        T = set threshold, if the error is smaller than T, the new points can be considered as new ininliers
        LOOP(A[0], A[len(A)-1]): run through all the points
            error = the distance between ground truth and the computed value based on the homography
            if error < T:
                ininliers += A[i]
        NUM = set the numbers threshold if the number of ininliers is higher than NUM jump out of the loop

        if len(ininliers) > NUM:
            break
        else:
            run_time -= 1

    H = get the best homography
    A, B = get the matching points from A and B based on the best homography model

    return A, B

