import cv2
import pytesseract
import matplotlib.pyplot as plt


# Görüntüyü okuma işlemi
gorsel = cv2.imread('bmw.jpg')
# Grayscale dönüşümünü görsele uygulama
gray_gorsel = cv2.cvtColor(gorsel, cv2.COLOR_BGR2GRAY)

def plot_images(gorsel1,gorsel2,baslik1="",baslik2=""):
    fig=plt.figure(figsize=[15,15])
    ax1=fig.add_subplot(121)#alt değer
    ax1.imshow(gorsel1,cmap="gray")
    ax1.set(xticks=[],yticks=[],title=baslik1)
    
    ax2=fig.add_subplot(122)
    ax2.imshow(gorsel2,cmap="gray")
    ax2.set(xticks=[],yticks=[],title=baslik2)

plot_images(gorsel,gray_gorsel,baslik1="Araç:BMW",baslik2="Araç:BMW")
blur = cv2.bilateralFilter(gray_gorsel, 11,90, 90)
plot_images(gray_gorsel, blur)
#Canny Kenar Tespiti
canny_edge = cv2.Canny(gray_gorsel, 170, 200)
# Kenarlar üzerinde konturları tanımlama
contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
# Plaka konturunu ve x, y koordinatlarını tanımlama
plaka_lisans_kontur = None
plaka_lisans = None
x = None
y = None
w = None
h = None
# 4 potansiyel köşeye sahip konturu bulun ve çevresindeki ROI(Seçim işlemi)'yi oluşturun
for contour in contours:
        # Kontur çevresini bulun ve kapalı bir kontur olmalıdır
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4: #Dikdörtgen olup olmadığına kontrol etme işlemi
            plaka_lisans_kontur = approx
            x, y, w, h = cv2.boundingRect(contour)
            plaka_lisans = gray_gorsel[y:y + h, x:x + w]
            break

# Tesseract'a göndermeden önce tespit edilen görüntüden gerekmeyen parçaların giderilmesi
plaka_lisans = cv2.bilateralFilter(plaka_lisans,11, 17, 17)
(thresh, plaka_lisans) = cv2.threshold(plaka_lisans, 150, 180, cv2.THRESH_BINARY)
#Metin Tanıma ve pytesseract yolu
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(plaka_lisans)
#plakayı çizme ve yazma
gorsel2 = cv2.rectangle(gorsel, (x,y), (x+w,y+h), (0,0,255), 3) 
gorsel2 = cv2.putText(gorsel, text, (x-20,y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

print("Araç Plakası :", text)

cv2.imshow("Arac Plaka Tanima",gorsel2)
cv2.imshow("Arac GrayScale",gray_gorsel)
blur = cv2.bilateralFilter(gray_gorsel, 11,90, 90)
plot_images(gray_gorsel, blur)
cv2.imshow("Arac Blur",blur)
#Canny Kenar Tespiti
canny_edge = cv2.Canny(gray_gorsel, 170, 200)
cv2.imshow("Arac Canny_Kenarlik_Cizimi",canny_edge)
cv2.waitKey(0)