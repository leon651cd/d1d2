f=open('d1_d2_qu_label')
i=0
precision_all=0
recall_all=0
f1_all=0
f_write=open('final_graph_result','a')
for line in f.readlines():
    i+=1
    if((i)%3==0):
        precision=line.strip().split()[2]
        recall=line.strip().split()[6]
        f1=line.strip().split()[10]
        precision=float(precision)
        recall=float(recall)
        f1=float(f1)
        precision_all+=precision
        recall_all+=recall
    if((i)%15==0):
        precision_all=precision_all/5
        recall_all=recall_all/5

        f_write.write("%.4f %.4f %.4f \n " % (precision_all,recall_all,2*precision_all*recall_all/(precision_all+recall_all)))
        precision_all=0
        recall_all=0
    else:
        continue




# a=   0.867257+ 0.845433+ 0.861432+ 0.857820+0.808156
# a=a/5
# b=   0.922705+ 0.895782+ 0.905558+ 0.898263+0.840744
# b=b/5
# result=2*a*b/(a+b)
# print(a)
# print(b)
# print(result)
#
#
#
# a=   0.888685+ 0.897499+ 0.953701
# a=a/3
# b=   0.888685+0.900128 +0.954138
# b=b/3
# result=2*a*b/(a+b)
# print(a)
# print(b)
# print(result)