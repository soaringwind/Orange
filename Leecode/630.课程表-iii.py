#
# @lc app=leetcode.cn id=630 lang=python3
#
# [630] 课程表 III
#

# @lc code=start
from typing import Coroutine, List
import heapq

class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        # last_time = [last for _, last in courses]
        # course_num = 0
        # course_day = 0
        # last_time.sort()
        # courses_sort = courses.copy()
        # res_list = []
        # # courses_sort_ = courses.copy()
        # for i in range(len(last_time)):
        #     last_time_ = last_time[i]
        #     for j in range(len(courses)):
        #         course = courses[j]
        #         if course[1]==last_time_:
        #             courses_sort[i] = course
        #             courses.pop(j)
        #             break
        # for i in range(len(courses_sort)):
        #     course = courses_sort[i]
        #     if course_day+course[0] <= course[1]:
        #         course_day += course[0]
        #         course_num += 1
        #         res_list.append(course[0])
        #     else:
        #         if res_list:
        #             if max(res_list) > course[0]:
        #                 idx = res_list.index(max(res_list))
        #                 course_day = course_day - res_list[idx] + course[0]
        #                 res_list.pop(idx)
        #                 res_list.append(course[0])
        # dur_time = [dur for dur, _ in courses_sort_]
        # course_num_ = 0
        # course_day_ = 0
        # dur_time.sort()
        # courses_sort = courses_sort_.copy()
        # for i in range(len(dur_time)):
        #     dur = dur_time[i]
        #     for j in range(len(courses_sort_)):
        #         course = courses_sort_[j]
        #         if course[0]==dur:
        #             courses_sort[i] = course
        #             courses_sort_.pop(j)
        #             break
        # for i in range(len(courses_sort)):
        #     course = courses_sort[i]
        #     if course_day_+course[0] <= course[1]:
        #         course_day_ += course[0]
        #         course_num_ += 1
        # return max(course_num, course_num_)
        courses.sort(key=lambda c: c[1])
        q = list()
        total = 0
        for ti, di in courses:
            if total + ti <= di:
                total += ti
                heapq.heappush(q, -ti)
            elif q and -q[0] > ti:
                total -= -q[0] - ti
                heapq.heappop(q)
                heapq.heappush(q, -ti)
        return len(q)

courses = [[5,5],[4,6],[2,6]]
print(Solution().scheduleCourse(courses))
        
# @lc code=end

