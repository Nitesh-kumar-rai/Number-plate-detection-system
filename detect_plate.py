import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
import pickle
#Write your image diectory in imread function
img=cv2.imread('crysta.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


gray_s=cv2.bilateralFilter(gray,11,17,17)

edged=cv2.Canny(gray_s,170,200)

cnts,new=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
img2=img.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)

cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
numberplatecnt=None
img2=img.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
img2=cv2.resize(img2,(600,300))

temp=0
for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,.02*peri,True)
    if len(approx)==4:
        numberplatecnt=approx
        x,y,w,h=cv2.boundingRect(c)
        temp=gray_s[y:y+h,x:x+w]
        temp=cv2.resize(temp,(600,300))
        break


img3=cv2.resize(temp,(600,300))

ret,img3 = cv2.threshold(temp,127,255,cv2.THRESH_BINARY)


dimension=img3.shape
size=dimension[0]*dimension[1]

img4=img3.copy()

char=[]
if size>1500:
    contours,hierarchy = cv2.findContours(img4,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    ins=0
    ind=[]
    ind_y=[]
    t=0
    for cnt in contours:
        if cnt.size>50:
            x,y,w,h = cv2.boundingRect(cnt)
            if h*w>700 and h>=w:
                for i in range(len(ind_y)):
                    if x>=max(ind_y):
                        ins=len(ind)
                        break
                    elif x+w<=min(ind):
                        ins=0
                        break
                    elif x>ind_y[i] and x+w<ind[i+1]:
                        ins=i+1
                        
                        break
                    elif x<ind[i] and x+w>ind_y[i]:
                        char.pop(i)
                        ind.pop(i)
                        ind_y.pop(i)
                        ins=i
                        break
                char.insert(ins,img4[y-2:y+h+2,x-2:x+w+2])
                ind.insert(ins,x)
                ind_y.insert(ins,x+w)


		


for i in range(len(char)):
    char[i]=cv2.resize(char[i],(50,50))

if size>1500:
	for i in range(len(char)):
		if ((int(char[i][1][1])+int(char[i][0][49])+int(char[i][49][0])+int(char[i][49][49]))/4)<127:
			char[i]=255-char[i]


file=open('lr_model.pickle','rb')

lr=pickle.load(file)



word={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:'A',11:'B',
      12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
      20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
      28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
      36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',
      44:'i',45:'j',46:'k',47:'l',48:'m',49:'n',50:'o',51:'p',
      52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',
      60:'y',61:'z'}

char_num=[]
for i in range(len(char)):
    test_X=char[i].reshape(1,-1)
    char_num.extend(lr.predict(test_X))



number=str()

for i in range(len(char_num)):
	number+=' '+str(word[char_num[i]])

number=number.upper().strip()
if number=='':
	print("Couldn't detect number")
else:
	print(number)
cv2.imshow(number,img3)
cv2.waitKey(0)
cv2.destroyAllWindows()



