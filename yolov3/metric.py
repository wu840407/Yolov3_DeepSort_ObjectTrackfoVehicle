import os

def read_file(file):
	with open(file,'r') as fp:
		temp = fp.read().strip('\n')
		line = temp.split('\n')
	for i in range(len(line)):
		line[i] = line[i].split(" ")
	return line

def iou(predict_bb,gt_bb):
	[x1,y1,x2,y2] = predict_bb
	[cx1,cy1,cx2,cy2] = gt_bb
	a = 0
	b = 1
	if not(float(x1) > float(cx2) or float(x2) < float(cx1) or float(y1) > float(cy2) or float(y2) < float(cy1)):
		[sx1,sx2,sx3,sx4] = sorted([float(x1),float(x2),float(cx1),float(cx2)])
		[sy1,sy2,sy3,sy4] = sorted([float(y1),float(y2),float(cy1),float(cy2)])
		a = (float(sx3) - float(sx2))*(float(sy3) - float(sy2))
		b = (float(x2) - float(x1))*(float(y2) - float(y1)) + (float(cx2) - float(cx1))*(float(cy2) - float(cy1)) - a
	return a/b

def AP(bb_confidence,bb_iou,gt_number,threshold = 0.5):
	fp,tp,ap = 0,0,0
	# sort iou
	x,y = [list(h) for h in zip(*sorted(zip(bb_confidence,bb_iou),reverse = True))] #x : sorted confidence, y : sorted iou
	r = 0
	p = 1
	for i in range(len(x)):
		if y[i] < threshold:
			fp += 1
		else:
			tp += 1
		new_r = tp/gt_number
		new_p = tp/(i+1)
		ap += 0.5*(p+new_p)*abs(r-new_r)
		r = new_r
		p = new_p
	return fp,tp,ap

def compute(clas,ground_truth,predict,threshold = 0.3):
	# initial
	fn,fp,tp,ap,flag = 0,0,0,0,0
	bb_confidence,bb_iou = [],[]
	all_gt_bb_list = []
	all_pr_bb_list = []

	for img in range(len(ground_truth)):
		# find class
		gt_bb_list = []
		for i in range(len(ground_truth[img])):
			if ground_truth[img][i][0] == clas:
				gt_bb_list.append(ground_truth[img][i])
		pr_bb_list = []
		for i in range(len(predict[img])):
			if predict[img][i][0] == clas:
				pr_bb_list.append(predict[img][i])
		all_gt_bb_list.extend(gt_bb_list)
		all_pr_bb_list.extend(pr_bb_list)
		# if gt and predict have data
		iou_table = [[-1 for i in range(len(pr_bb_list))] for j in range(len(gt_bb_list))]
		for i in range(len(gt_bb_list)):
			gt_bb = [float(gt_bb_list[i][1])-float(gt_bb_list[i][3])/2,float(gt_bb_list[i][2])-float(gt_bb_list[i][4])/2,float(gt_bb_list[i][1])+float(gt_bb_list[i][3])/2,float(gt_bb_list[i][2])+float(gt_bb_list[i][4])/2]
			for j in range(len(pr_bb_list)):
				predict_bb = [float(pr_bb_list[j][2])-float(pr_bb_list[j][4])/2,float(pr_bb_list[j][3])-float(pr_bb_list[j][5])/2,float(pr_bb_list[j][2])+float(pr_bb_list[j][4])/2,float(pr_bb_list[j][3])+float(pr_bb_list[j][5])/2]
				iou_table[i][j] = iou(predict_bb,gt_bb)
		## calculate FN
		for i in range(len(iou_table)):
			if all([ x<threshold for x in iou_table[i]]) :
				# gt isn't predict
				fn += 1
			else:
				for j in range(len(iou_table[i])):
					if iou_table[i][j] >= threshold :
						bb_confidence.append(pr_bb_list[j][1])
						bb_iou.append(iou_table[i][j])

	# if no gt or no predict in this class
	if all_pr_bb_list and all_gt_bb_list :
		fp,tp,ap = AP(bb_confidence,bb_iou,len(all_gt_bb_list))
		flag = 1
	elif all_pr_bb_list and all_gt_bb_list == []:
		fp += len(pr_bb_list)
		flag = 1
	elif all_pr_bb_list == [] and all_gt_bb_list:
		fn += len(gt_bb_list)
		flag = 1
	return fn,fp,tp,ap,flag

## read file gt : [img][bb number][class,x,y,w,h] pr : [img][bb number][class,confidence,x,y,w,h]
path = os.getcwd()
ground_truth_list = os.listdir(os.path.join(path,".\data\car\labels\\27"))
predict_list = os.listdir(os.path.join(path,".\output"))
ground_truth_list.sort()
predict_list.sort()

ground_truth,predict = [],[]
for gt in ground_truth_list:
	ground_truth.append(read_file(os.path.join(path,".\data\car\labels\\27",gt)))
for pr in predict_list:
	predict.append(read_file(os.path.join(path,".\output",pr)))

## 36 classes
FN,FP,TP,SAP = 0,0,0,0
category = 0

for i in range(36):
	fn,fp,tp,ap,flag = compute(str(i),ground_truth,predict)
	#print(fn,fp,tp,ap)
	TP += tp
	FN += fn
	FP += fp
	SAP += ap
	if flag == 1:
		category += 1
	flag = 0

print("TP : ",TP)
print("FP : ",FP)
print("FN : ",FN)
if FN+FP+TP == 0:
	print("Accuracy : ",0)
else:
	print("Accuracy : ",TP/(FN+FP+TP))
print("mAP : ",SAP/category)