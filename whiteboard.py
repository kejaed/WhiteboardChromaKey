import numpy as np
import cv2

# frame to start looking at 
#startFrame = 0
startFrame = 5730

# number of frames to skip each time
skipFrames = 1

# show the movie and plot metrics or not (much faster when not)
showGUI = 0 ;
aimport numpy as np
import cv2

# frame to start looking at 
#startFrame = 0
startFrame = 6000

# number of frames to skip each time
skipFrames = 1

# show the movie and plot metrics or not (much faster when not)
showGUI = 0 ;

cap = cv2.VideoCapture('sd.mp4')

cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, startFrame); 

numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT);


ret,frame = cap.read()
cv2.imwrite("frame.png",frame)

TL = (187,43)
TR = (502,28)
BL = (165,260)
BR = (475,285)

TR_BL = np.subtract(TR , BL)
TR_TL = np.subtract(TR , TL)
TR_BR = np.subtract(TR , BR)

imsize =  frame.shape
print imsize
print TR_BL 
print TR_TL 
print TR_BR 

# some notes
# we can use the top right edge to see if the screen has moved, that isn't
# ever really blocked by jordan

# so let's take that as a template and just match to that
# TODO need to upload the right one to github
template = cv2.imread('top_right.png')
Th,Tw,d = template.shape
#print h,w,d

wbContent = cv2.imread('wb_text.png')

while (1):
    
    ret,frame = cap.read()
    curFrame = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES));     
    print 'Frame ', curFrame
    
    # see if we got a good frame
    if ret == True:

        # we have a template of the top right of the board, find the best match in the frame
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(frame,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #in our case, if max_val > .70 we have the right scene and really did find the whiteboard
        if max_val > .70:

            # 56 and 36 are from edge of template to corner of whiteboard in that template
            TR = (max_loc[0] + 56, max_loc[1] + 32)

            # fin the other corners by hand-tuned offsets
            TL = tuple(TR - TR_TL)
            BL = tuple(TR - TR_BL)
            BR = tuple(TR - TR_BR)
            
            #cv2.rectangle(frame,max_loc, Tbottom_right, 255, 2)

            if 0:
                cv2.circle(frame , TL, 2,(0,0,255),-1)
                cv2.circle(frame , TR, 2,(0,0,255),-1)
                cv2.circle(frame , BL, 2,(0,0,255),-1)
                cv2.circle(frame , BR, 2,(0,0,255),-1)

                cv2.line(frame,TL,TR,(0,0,255),2)
                cv2.line(frame,TR,BR,(0,0,255),2)
                cv2.line(frame,BR,BL,(0,0,255),2)
                cv2.line(frame,BL,TL,(0,0,255),2)

            # using http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

            rect = np.zeros((4, 2), dtype = "float32")
            rect[0] = TL
            rect[1] = TR
            rect[2] = BR
            rect[3] = BL

            # how the URL guessed at size of the board
            #widthA = np.sqrt(((BR[0] - BL[0]) ** 2) + ((BR[1] - BL[1]) ** 2))
            #widthB = np.sqrt(((TR[0] - TL[0]) ** 2) + ((TR[1] - TL[1]) ** 2))
            #maxWidth = max(int(widthA), int(widthB))

            #heightA = np.sqrt(((TR[0] - BR[0]) ** 2) + ((TR[1] - BR[1]) ** 2))
            #heightB = np.sqrt(((TL[0] - BL[0]) ** 2) + ((TL[1] - BL[1]) ** 2))
            #maxHeight = max(int(heightA), int(heightB))

            # set the height in pixels of our virtual whiteboard
            # let's try 16x9            
            # TODO determine the right virtual whiteboard size
            wbWidth  = 800
            wbHeight = 450
            
            dst = np.array([
		[0, 0],
		[wbWidth - 1, 0],
		[wbWidth - 1, wbHeight - 1],
		[0, wbHeight - 1]], dtype = "float32")

            # create our transformation matrices
            
            # from frame to virtual whitebaord
            M = cv2.getPerspectiveTransform(rect, dst)

            # from virtual whitebaord to frame
            Minv = cv2.getPerspectiveTransform(dst, rect)

            # create our virtual whiteboard
            virtualWhiteboard = cv2.warpPerspective(frame, M, (wbWidth, wbHeight))

            # draw a blue rectangle on virtual whiteboard to test
            # cv2.rectangle(virtualWhiteboard, (wbWidth/3,wbHeight/3), (wbWidth*2/3,wbHeight/2), 255, -1)
            
            #cv2.imshow('virtualWhiteboard',virtualWhiteboard)

            # create forground / background masks
            # the background is the whiteboard we are going to replace

            # let's just try a threshold first, then we need to take care of the parts that are
            # part of Jordan's shirt
            img2gray = cv2.cvtColor(virtualWhiteboard,cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(img2gray,100,200)
            cv2.imshow('edges',edges)

            THRESHOLD = 240 ; 

            ret, mask = cv2.threshold(img2gray, THRESHOLD, 255, cv2.THRESH_BINARY)
            #gray = cv2.GaussianBlur(img2gray,(5,5),0)
            #ret, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask_inv = cv2.bitwise_not(mask)
            cv2.imshow('mask',mask)
            

            contours, hierarchy = cv2.findContours(mask_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            
            # as fun as the convex hull is, it's not what we want here.
            #hull = cv2.convexHull(cnt)
            #cv2.drawContours(virtualWhiteboard, [hull], -1, (0,255,0), 3)

            #cv2.drawContours(virtualWhiteboard, [cnt], -1, (0,255,255), -1)

    
            jordanMask = np.zeros(img2gray.shape,np.uint8)
            allWhite   = np.ones(img2gray.shape,np.uint8)
            allWhite = np.multiply(allWhite,255)
            cv2.drawContours(jordanMask,[cnt],0,255,-1)

            whiteboardMask = cv2.bitwise_not(jordanMask)

            wbContentMasked = cv2.bitwise_and(wbContent,wbContent,mask = whiteboardMask)
            jordanMasked = cv2.bitwise_and(virtualWhiteboard,virtualWhiteboard,mask = jordanMask)

            virtualWhiteboard = cv2.add(wbContentMasked,jordanMasked)

            #cv2.imshow('wbContent',wbContentMasked)
            #cv2.imshow('jordanMasked',jordanMasked)
            #cv2.imshow('virtualWhiteboard',virtualWhiteboard)
            

            

            

    
            # go from virtual whiteboard to our frame
            unWarped = cv2.warpPerspective(virtualWhiteboard, Minv, (imsize[1], imsize[0]),flags=cv2.INTER_NEAREST)

            unWarpedMask = cv2.warpPerspective(allWhite, Minv, (imsize[1], imsize[0]),flags=cv2.INTER_NEAREST)

            # before we unwarp, make the mask a little smaller than our actual size
            # to remove any jaggies
            kernel = np.ones((5,5),np.uint8)
            unWarpedMask2 = cv2.erode(unWarpedMask,kernel,iterations = 1)
            
            unWarpedMaskInv = cv2.bitwise_not(unWarpedMask2)
            frameMask = cv2.bitwise_and(frame,frame,mask = unWarpedMaskInv)

            # update the mask on unWarped
            unWarped2 = cv2.bitwise_and(unWarped,unWarped,mask = unWarpedMask2)
            frameNew = cv2.add(frameMask,unWarped2)
            
            cv2.imshow('unWarped',unWarped)
            cv2.imshow('frameNew',frameNew)

            
            
        else:
            print 'Didnt find template'    

            
        
        
        
       # cv2.imshow('Original',frame)
        key = cv2.waitKey(100)
        if key  == 27: #escape
            print "exiting!"
            break
    else:
        print 'done with movie'
        break

    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, curFrame + skipFrames); 


print 'Done'
cv2.destroyAllWindows()

cap = cv2.VideoCapture('sd.mp4')

cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, startFrame); 

numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT);


ret,frame = cap.read()
cv2.imwrite("frame.png",frame)

TL = (187,43)
TR = (502,28)
BL = (165,260)
BR = (475,285)

TR_BL = np.subtract(TR , BL)
TR_TL = np.subtract(TR , TL)
TR_BR = np.subtract(TR , BR)

imsize =  frame.shape
print imsize
print TR_BL 
print TR_TL 
print TR_BR 

# some notes
# we can use the top right edge to see if the screen has moved, that isn't
# ever really blocked by jordan

# so let's take that as a template and just match to that
# TODO need to upload the right one to github
template = cv2.imread('top_right.png')
Th,Tw,d = template.shape
#print h,w,d

nature = cv2.imread('sully.jpg')

while (1):
    
    ret,frame = cap.read()
    curFrame = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES));     
    print 'Frame ', curFrame
    
    # see if we got a good frame
    if ret == True:

        # we have a template of the top right of the board, find the best match in the frame
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(frame,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #in our case, if max_val > .70 we have the right scene and really did find the whiteboard
        if max_val > .70:

            # 56 and 36 are from edge of template to corner of whiteboard in that template
            TR = (max_loc[0] + 56, max_loc[1] + 32)

            # fin the other corners by hand-tuned offsets
            TL = tuple(TR - TR_TL)
            BL = tuple(TR - TR_BL)
            BR = tuple(TR - TR_BR)
            
            #cv2.rectangle(frame,max_loc, Tbottom_right, 255, 2)

            if 0:
                cv2.circle(frame , TL, 2,(0,0,255),-1)
                cv2.circle(frame , TR, 2,(0,0,255),-1)
                cv2.circle(frame , BL, 2,(0,0,255),-1)
                cv2.circle(frame , BR, 2,(0,0,255),-1)

                cv2.line(frame,TL,TR,(0,0,255),2)
                cv2.line(frame,TR,BR,(0,0,255),2)
                cv2.line(frame,BR,BL,(0,0,255),2)
                cv2.line(frame,BL,TL,(0,0,255),2)

            # using http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

            rect = np.zeros((4, 2), dtype = "float32")
            rect[0] = TL
            rect[1] = TR
            rect[2] = BR
            rect[3] = BL

            # how the URL guessed at size of the board
            #widthA = np.sqrt(((BR[0] - BL[0]) ** 2) + ((BR[1] - BL[1]) ** 2))
            #widthB = np.sqrt(((TR[0] - TL[0]) ** 2) + ((TR[1] - TL[1]) ** 2))
            #maxWidth = max(int(widthA), int(widthB))

            #heightA = np.sqrt(((TR[0] - BR[0]) ** 2) + ((TR[1] - BR[1]) ** 2))
            #heightB = np.sqrt(((TL[0] - BL[0]) ** 2) + ((TL[1] - BL[1]) ** 2))
            #maxHeight = max(int(heightA), int(heightB))

            # set the height in pixels of our virtual whiteboard
            # let's try 16x9            
            # TODO determine the right virtual whiteboard size
            wbWidth  = 800
            wbHeight = 450
            
            dst = np.array([
		[0, 0],
		[wbWidth - 1, 0],
		[wbWidth - 1, wbHeight - 1],
		[0, wbHeight - 1]], dtype = "float32")

            # create our transformation matrices
            
            # from frame to virtual whitebaord
            M = cv2.getPerspectiveTransform(rect, dst)

            # from virtual whitebaord to frame
            Minv = cv2.getPerspectiveTransform(dst, rect)

            # create our virtual whiteboard
            virtualWhiteboard = cv2.warpPerspective(frame, M, (wbWidth, wbHeight))

            # draw a blue rectangle on virtual whiteboard to test
            # cv2.rectangle(virtualWhiteboard, (wbWidth/3,wbHeight/3), (wbWidth*2/3,wbHeight/2), 255, -1)
            
            #cv2.imshow('virtualWhiteboard',virtualWhiteboard)

            # create forground / background masks
            # the background is the whiteboard we are going to replace

            # let's just try a threshold first, then we need to take care of the parts that are
            # part of Jordan's shirt
            img2gray = cv2.cvtColor(virtualWhiteboard,cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(img2gray,100,200)
            cv2.imshow('edges',edges)

            THRESHOLD = 220 ; 

            ret, mask = cv2.threshold(img2gray, THRESHOLD, 255, cv2.THRESH_BINARY)
            #gray = cv2.GaussianBlur(img2gray,(5,5),0)
            #ret, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask_inv = cv2.bitwise_not(mask)
            cv2.imshow('mask',mask)
            

            contours, hierarchy = cv2.findContours(mask_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            
            # as fun as the convex hull is, it's not what we want here.
            #hull = cv2.convexHull(cnt)
            #cv2.drawContours(virtualWhiteboard, [hull], -1, (0,255,0), 3)

            #cv2.drawContours(virtualWhiteboard, [cnt], -1, (0,255,255), -1)

    
            jordanMask = np.zeros(img2gray.shape,np.uint8)
            allWhite   = np.ones(img2gray.shape,np.uint8)
            allWhite = np.multiply(allWhite,255)
            cv2.drawContours(jordanMask,[cnt],0,255,-1)

            whiteboardMask = cv2.bitwise_not(jordanMask)

            natureMasked = cv2.bitwise_and(nature,nature,mask = whiteboardMask)
            jordanMasked = cv2.bitwise_and(virtualWhiteboard,virtualWhiteboard,mask = jordanMask)

            virtualWhiteboard = cv2.add(natureMasked,jordanMasked)

            #cv2.imshow('nature',natureMasked)
            #cv2.imshow('jordanMasked',jordanMasked)
            #cv2.imshow('virtualWhiteboard',virtualWhiteboard)
            

            

            

    
            # go from virtual whiteboard to our frame
            unWarped = cv2.warpPerspective(virtualWhiteboard, Minv, (imsize[1], imsize[0]),flags=cv2.INTER_NEAREST)

            unWarpedMask = cv2.warpPerspective(allWhite, Minv, (imsize[1], imsize[0]),flags=cv2.INTER_NEAREST)

            # before we unwarp, make the mask a little smaller than our actual size
            # to remove any jaggies
            kernel = np.ones((5,5),np.uint8)
            unWarpedMask2 = cv2.erode(unWarpedMask,kernel,iterations = 1)
            
            unWarpedMaskInv = cv2.bitwise_not(unWarpedMask2)
            frameMask = cv2.bitwise_and(frame,frame,mask = unWarpedMaskInv)

            # update the mask on unWarped
            unWarped2 = cv2.bitwise_and(unWarped,unWarped,mask = unWarpedMask2)
            frameNew = cv2.add(frameMask,unWarped2)
            
            cv2.imshow('unWarped',unWarped)
            cv2.imshow('frameNew',frameNew)

            
            
        else:
            print 'Didnt find template'    

            
        
        
        
       # cv2.imshow('Original',frame)
        key = cv2.waitKey(100)
        if key  == 27: #escape
            print "exiting!"
            break
    else:
        print 'done with movie'
        break

    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, curFrame + skipFrames); 


print 'Done'
cv2.destroyAllWindows()
