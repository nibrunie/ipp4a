import math

poi_list = [(59.875, 136.875, 8.0), (11.882353, 288.82352, 17.0), (19.181818, 361.45456, 11.0), (123.2, 35.599998, 5.0), (112.62069, 328.96552, 29.0), (114.16666, 521.83331, 6.0), (78.5, 612.0, 42.0), (125.0, 723.41669, 12.0), (203.875, 33.875, 8.0), (188.0, 160.0, 13.0), (187.88889, 165.11111, 9.0), (211.21428, 276.64285, 14.0), (156.0, 578.125, 16.0), (232.74074, 88.259262, 27.0), (218.0, 276.0, 5.0), (253.5, 283.0, 22.0), (218.5, 421.0, 10.0), (270.71429, 632.5, 14.0), (223.78572, 700.07141, 14.0), (222.90909, 704.63635, 11.0), (228.0, 847.5, 24.0), (310.71429, 132.89285, 28.0), (333.95001, 185.75, 20.0), (325.0, 759.5, 8.0), (362.36365, 47.772728, 22.0), (377.95999, 48.240002, 25.0), (363.63635, 55.0, 11.0), (410.09091, 86.909088, 22.0), (417.66666, 165.08333, 12.0), (368.0, 220.0, 9.0), (408.18518, 421.18518, 27.0), (362.73077, 473.23077, 26.0), (385.0, 693.0, 19.0), (502.57144, 32.714287, 7.0), (490.69696, 56.545456, 33.0), (488.04999, 191.25, 20.0), (442.3913, 310.08694, 23.0), (471.0, 272.5, 6.0), (470.86206, 297.10345, 29.0), (458.0, 339.16666, 6.0), (455.41666, 414.58334, 12.0), (479.15384, 386.38461, 13.0), (494.0, 417.0, 19.0), (433.0, 435.5, 8.0), (436.60001, 432.79999, 5.0), (438.25, 453.875, 8.0), (443.61111, 451.16666, 18.0), (443.875, 519.0, 16.0), (490.11111, 492.88889, 18.0), (435.66666, 550.66669, 12.0), (449.7027, 736.18915, 37.0), (477.45834, 774.875, 24.0), (506.30768, 32.153847, 13.0), (535.27271, 29.727272, 11.0), (523.16669, 103.72222, 18.0), (557.8125, 104.375, 16.0), (523.0, 110.0, 5.0), (557.0, 110.5, 6.0), (537.76471, 207.52942, 17.0), (504.625, 354.0, 8.0), (505.78262, 381.13043, 23.0), (522.84613, 412.53845, 13.0), (533.94739, 379.73685, 19.0), (520.0, 453.5, 12.0), (515.33331, 498.0, 9.0), (529.45453, 572.18182, 11.0), (562.33331, 555.33331, 6.0), (531.33331, 690.41669, 24.0), (539.0, 681.75, 16.0), (556.0, 861.0, 5.0), (610.15387, 2.3846154, 13.0), (583.03333, 65.26667, 30.0), (592.85712, 101.0, 7.0), (628.75, 68.0, 16.0), (576.45453, 121.72727, 11.0), (577.69232, 435.23077, 13.0), (580.0, 435.0, 7.0), (641.20001, 486.60001, 5.0), (641.90906, 522.72723, 22.0), (578.625, 707.1875, 16.0), (678.59998, 12.2, 5.0), (682.0, 106.5, 6.0), (681.63635, 110.63636, 11.0), (690.66669, 124.66666, 6.0), (701.8125, 171.375, 16.0), (660.85712, 225.85715, 7.0), (651.16669, 407.0, 6.0), (649.875, 442.875, 8.0), (671.875, 670.25, 8.0), (666.9375, 709.09375, 32.0), (676.21875, 732.65625, 32.0), (704.25, 782.04999, 20.0), (650.5, 848.0, 6.0), (788.44116, 165.58824, 34.0), (752.375, 241.375, 8.0), (756.0, 241.60001, 5.0), (723.0, 377.0, 7.0), (723.0, 382.0, 9.0), (742.15387, 396.38461, 13.0), (778.0, 467.0, 7.0), (774.0, 557.0, 7.0), (740.125, 675.125, 8.0), (786.0, 833.0, 9.0), (794.5, 165.0, 6.0), (822.29633, 326.33334, 27.0), (829.30768, 335.38461, 13.0), (798.75, 413.17856, 28.0), (797.53845, 568.61542, 13.0), (805.875, 555.0, 16.0), (850.45453, 603.18182, 11.0), (837.90002, 837.95001, 20.0), (922.45831, 8.666667, 24.0), (926.0, 117.5, 36.0), (892.66669, 220.88889, 9.0), (883.0, 434.0, 7.0), (864.875, 688.875, 8.0), (913.91492, 727.48938, 47.0), (917.02631, 786.23682, 38.0), (938.57892, 91.473686, 19.0), (940.0, 432.0, 9.0), (955.0, 479.72223, 18.0), (987.11108, 445.88889, 18.0), (948.06451, 672.16132, 31.0), (957.0, 679.35712, 14.0), (963.09088, 729.90906, 22.0), (977.2121, 720.45453, 33.0), (941.0, 759.0, 9.0), (1015.4, 152.48, 25.0), (1060.7142, 129.80952, 21.0), (1037.5, 457.5, 14.0), (1146.8889, 406.22223, 9.0)]

poi_list2 = [(58.200001, 142.0, 10.0), (10.117647, 294.41177, 17.0), (17.583334, 367.25, 12.0), (111.13333, 334.53333, 30.0), (112.8, 527.40002, 5.0), (77.048782, 617.85364, 41.0), (123.45454, 729.18182, 11.0), (202.0, 39.714287, 7.0), (186.5, 166.28572, 14.0), (209.5, 282.5, 14.0), (215.0, 434.0, 5.0), (154.53847, 583.61542, 13.0), (231.5, 94.0, 26.0), (251.95, 288.75, 20.0), (241.8, 342.39999, 5.0), (218.86667, 426.86667, 15.0), (219.5, 433.83334, 12.0), (269.33334, 638.33331, 12.0), (222.05263, 706.36841, 19.0), (226.59091, 853.04547, 22.0), (358.20001, 56.900002, 10.0), (309.41379, 138.44827, 29.0), (332.33334, 191.66667, 18.0), (358.66666, 479.5, 6.0), (362.63635, 52.090908, 11.0), (377.0, 52.625, 8.0), (362.3158, 56.315788, 19.0), (376.36841, 56.0, 19.0), (408.75, 92.75, 20.0), (415.875, 167.875, 8.0), (406.76923, 426.92307, 26.0), (362.04999, 478.85001, 20.0), (363.33334, 486.88889, 9.0), (383.52942, 699.11768, 17.0), (383.0, 705.0, 7.0), (501.90909, 38.363636, 11.0), (488.59259, 62.148148, 27.0), (492.39999, 57.799999, 5.0), (486.78946, 197.05263, 19.0), (440.80951, 316.04761, 21.0), (469.56668, 302.70001, 30.0), (457.0, 345.0, 7.0), (454.0, 420.5, 10.0), (477.61539, 392.38461, 13.0), (492.36841, 423.05264, 19.0), (436.0, 434.0, 9.0), (442.46667, 524.66675, 15.0), (458.42856, 494.57144, 7.0), (488.75, 498.5625, 16.0), (434.45456, 556.18182, 11.0), (448.1842, 742.05261, 38.0), (475.95834, 780.70831, 24.0), (507.0, 38.0, 7.0), (533.75, 35.416668, 12.0), (571.0, 1.0, 9.0), (573.0, 70.0, 5.0), (521.95001, 109.75, 20.0), (556.23077, 110.23077, 13.0), (536.40002, 213.39999, 15.0), (557.125, 181.5, 8.0), (510.15384, 378.46155, 13.0), (505.36365, 383.90909, 11.0), (518.38464, 459.46155, 13.0), (514.0, 503.66666, 9.0), (528.33331, 578.0, 9.0), (561.0, 561.0, 7.0), (530.14288, 696.14288, 21.0), (537.5, 687.5, 14.0), (570.0, 861.0, 5.0), (608.66669, 8.25, 12.0), (581.50006, 70.999985, 28.0), (591.0, 106.16666, 6.0), (627.26666, 73.933334, 15.0), (580.17645, 432.58823, 17.0), (589.34998, 435.0, 20.0), (640.0, 492.0, 7.0), (640.72223, 528.83331, 18.0), (577.375, 712.8125, 16.0), (576.75, 861.0, 8.0), (677.16669, 17.833334, 6.0), (680.33331, 113.0, 9.0), (689.59998, 130.2, 5.0), (700.28571, 177.35715, 14.0), (660.4375, 232.0, 16.0), (648.71429, 448.42856, 7.0), (701.70001, 433.70001, 10.0), (670.125, 676.125, 8.0), (665.64514, 714.93549, 31.0), (673.0, 704.5, 6.0), (674.89655, 738.51721, 29.0), (702.90002, 787.95001, 20.0), (650.0, 854.0, 5.0), (745.20001, 83.199997, 5.0), (787.22858, 171.45714, 35.0), (720.0, 381.0, 7.0), (740.61536, 402.38461, 13.0), (777.0, 472.71429, 7.0), (772.71429, 562.85712, 7.0), (739.0, 681.0, 7.0), (742.20001, 714.59998, 5.0), (784.75, 838.625, 8.0), (794.0, 171.0, 5.0), (806.0, 326.5, 6.0), (797.69232, 418.96155, 26.0), (796.18182, 574.45453, 11.0), (804.35712, 560.78577, 14.0), (849.125, 609.125, 8.0), (836.58826, 843.88232, 17.0), (921.08698, 14.391304, 23.0), (934.57141, 97.714287, 7.0), (924.71875, 123.3125, 32.0), (891.5, 226.5, 8.0), (934.57141, 429.0, 7.0), (882.0, 439.71429, 7.0), (934.0, 435.0, 7.0), (867.08331, 694.91669, 12.0), (867.90002, 702.90002, 10.0), (912.97827, 733.43475, 46.0), (915.88574, 792.11426, 35.0), (938.20001, 96.933334, 15.0), (939.48151, 433.55554, 27.0), (954.0, 484.625, 8.0), (986.06665, 451.73334, 15.0), (953.71429, 488.21429, 14.0), (946.62067, 677.93103, 29.0), (956.0, 685.21429, 14.0), (961.80951, 735.85712, 21.0), (975.96667, 726.40002, 30.0), (939.875, 764.875, 8.0), (1014.2174, 158.26086, 23.0), (1059.5, 135.5, 18.0), (1014.0, 164.5, 6.0), (1036.6154, 463.38461, 13.0), (1015.56, 750.79999, 25.0), (1010.0, 801.0, 5.0), (1010.0, 810.0, 5.0), (1145.8, 412.0, 10.0)]

poi_list.sort(key = lambda v: v[2])
poi_list2.sort(key = lambda v: v[2])
print(poi_list[-10:])
print(poi_list2[-10:])
print("list0: {}".format(len(poi_list)))
print("list1: {}".format(len(poi_list2)))


def exp_weight(v0, v1):
  return math.exp(abs(v0[2] - v1[2]))

def distance(p0, p1, weight = (lambda v0, v1: 1.0)):
  dx = (p0[0] - p1[0]) **2
  dy = (p0[1] - p1[1]) **2
  return math.sqrt(dx + dy) * weight(p0, p1)

# erase points that are too close, as they may be duplicate
def delete2Close(point_list, threshold = 5.0):
  result = []
  # computing distance matrix
  for i in xrange(len(point_list)):
    add = True
    for j in xrange(len(point_list)):
      if i != j and distance(point_list[i], point_list[j], exp_weight) < threshold:
        add = False
        break
    if add: 
      result.append(point_list[i])
  return result
      
print("deleting too close neighbours") 
list0 = delete2Close(poi_list, 20.0)
list1 = delete2Close(poi_list2, 20.0)
print("list0: {}, {}".format(len(list0), list0[-5:]))
print("list1: {}, {}".format(len(list1), list1[-5:]))

for a, b in zip(list0[-5:], list1[-5:]):
  print distance(a, b)

class Mat2x2:
  def __init__(self, _00, _01, _10, _11):
    self._00 = _00
    self._01 = _01
    self._10 = _10
    self._11 = _11

#list0 = list0[-10:]
#list1 = list1[-10:]

# compute the list of pairs of closest from 
# list0 and list1
def compute_pairs(list0, list1, threshold = 10.0):
  result = []
  valid_list1 = [i for i in xrange(len(list1))]
  for p0 in list0:
    ordered_list = [(i, distance(p0, list1[i])) for i in valid_list1]
    ordered_list.sort(key = lambda v: v[1])
    elected, d = ordered_list[0]
    if d < threshold: result.append((p0, list1[elected]))
    valid_list1.remove(elected)
  return result

for p0, p1 in compute_pairs(list0, list1):
    print(distance(p0, p1))
average = sum(distance(p0, p1) for p0, p1 in compute_pairs(list0, list1)) / len(list0)
print("average: {}".format(average))

class Transform:
  def __init__(self, b, A, db, dA):
    self.b = b
    self.A = A
    self.db = db
    self.dA = dA

  def apply(self, v):
    return (self.b[0] + self.A[0][0] * v[0] + self.A[0][1] * v[1], 
            self.b[1] + self.A[1][0] * v[0] + self.A[1][1] * v[1],
            v[2])

def icp_iteration(ref_list, moved_list, tfm):
  closest_pairs = compute_pairs(ref_list, moved_list)

  dG_dbx = sum((2  *  (p0[0] - p1[0])) for (p0, p1) in closest_pairs)
  dG_dby = sum((2  *  (p0[1] - p1[1])) for (p0, p1) in closest_pairs)
  dG_dA00 = sum(2  *  p0[0] * (p0[0] - p1[0]) for (p0, p1) in closest_pairs)
  dG_dA01 = sum(2  *  p0[1] * (p0[0] - p1[0]) for (p0, p1) in closest_pairs)
  dG_dA10 = sum(2  *  p0[0] * (p0[1] - p1[1]) for (p0, p1) in closest_pairs)
  dG_dA11 = sum(2  *  p0[1] * (p0[1] - p1[1]) for (p0, p1) in closest_pairs)

  # goal is to minimize G so to update in the direction
  # it decreases <=> dG_dV is negative
  dbx = tfm.db if dG_dbx > 0 else -tfm.db
  dby = tfm.db if dG_dby > 0 else -tfm.db

  dA00 = tfm.dA if dG_dA00 > 0 else -tfm.dA
  dA01 = tfm.dA if dG_dA01 > 0 else -tfm.dA
  dA10 = tfm.dA if dG_dA10 > 0 else -tfm.dA
  dA11 = tfm.dA if dG_dA11 > 0 else -tfm.dA

  tfm.b[0]    += dbx 
  tfm.b[1]    += dby 
  tfm.A[0][0] += dA00
  tfm.A[0][1] += dA01
  tfm.A[1][0] += dA10
  tfm.A[1][1] += dA11
  return tfm

transform = Transform([0,-6],[[1,0], [0,1]], 0.1, 0.0)
ref_list = list0
src_list = list1

iteration_scheme = [(0.1, 0.0, 1000), (0.001, 0.000, 1000)]

for db, dA, num_iteration in iteration_scheme:
  transform.db = db
  transform.dA = dA

  for i in xrange(num_iteration):
    moved_list = [transform.apply(v) for v in src_list]
    closest_pairs = compute_pairs(ref_list, moved_list)
    #print("len(closest_pairs)={}".format(len(closest_pairs)))
    #d_list = [distance(u,v) for u,v in closest_pairs]
    #d_list.sort()
    #G = sum(d_list)
    #print d_list[:10]
    #print("G={}\n".format(G))
    transform = icp_iteration(ref_list, moved_list, transform)

moved_list = [transform.apply(v) for v in src_list]
average = sum(distance(p0, p1) for p0, p1 in compute_pairs(ref_list, moved_list)) / len(ref_list)
closest_pairs =  compute_pairs(list0, moved_list)
d_list = [distance(u,v) for u,v in closest_pairs]
d_list.sort()
print d_list[:10]
print("average: {}".format(average))
print("b=({},{}) A=(({},{}),({},{}))".format(transform.b[0], transform.b[1], transform.A[0][0], transform.A[0][1], transform.A[1][0], transform.A[1][1]))