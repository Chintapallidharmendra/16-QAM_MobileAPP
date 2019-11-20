import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager ,Screen
from kivy.uix.image import Image
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout

import matplotlib.pyplot as plt
import numpy as np
import math

class WindowManager(ScreenManager):
    pass

class MainWindow(Screen):
    bitin = ObjectProperty(None)
    mval = ObjectProperty(None)
    
    def btn(self):
        #print("Bit_input:", self.bitin.text, "M_input:", self.mval.text)
        QAMmod(self.bitin.text, self.mval.text)
        self.bitin.text = ""
        self.mval.text = ""
        

class FirstWindow(Screen):
    inbit = ObjectProperty(None)

    def btn1(self):
        self.ids.inbit.source = "input.png"
        self.ids.inbit.reload()
   

class SecondWindow(Screen):
    modu = ObjectProperty(None)

    def btn2(self):
        self.ids.modu.source = "modulated.png"
        self.ids.modu.reload()
    

class ThirdWindow(Screen):
    demodu = ObjectProperty(None)

    def btn3(self):
        self.ids.demodu.source = "demodulated.png"
        self.ids.demodu.reload()
    

class FourthWindow(Screen):
    cons = ObjectProperty(None)

    def btn4(self):
        self.ids.cons.source = "constellation.png"
        self.ids.cons.reload()
    

class FifthWindow(Screen):
    ffti = ObjectProperty(None)

    def btn5(self):
        self.ids.ffti.source = "fft.png"
        self.ids.ffti.reload()
    

kv = Builder.load_file("my.kv")
sm = WindowManager()

screens = [MainWindow(name="main"), FirstWindow(name="first"),SecondWindow(name="second"),
           ThirdWindow(name="third"),FourthWindow(name="fourth"),FifthWindow(name="fifth")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "main"

class MyApp(App):
    def build(self):
        return sm

def QAMmod(bitin,mval):
    mapping_table = {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -3-1j,
        (0,0,1,0) : -3+3j,
        (0,0,1,1) : -3+1j,
        (0,1,0,0) : -1-3j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : -1+3j,
        (0,1,1,1) : -1+1j,
        (1,0,0,0) :  3-3j,
        (1,0,0,1) :  3-1j,
        (1,0,1,0) :  3+3j,
        (1,0,1,1) :  3+1j,
        (1,1,0,0) :  1-3j,
        (1,1,0,1) :  1-1j,
        (1,1,1,0) :  1+3j,
        (1,1,1,1) :  1+1j
    }

    phase_table = {
        1 : np.pi + math.atan(1),
        2 : np.pi + math.atan(1/3),
        3 : np.pi - math.atan(1),
        4 : np.pi - math.atan(1/3),
        5 : np.pi + math.atan(3),
        6 : np.pi - math.atan(3),
        7 : 2*np.pi - math.atan(1),
        8 : 2*np.pi - math.atan(1/3),
        9 : math.atan(1),
        10 : math.atan(1/3),
        11 : 2*np.pi - math.atan(3),
        12 : math.atan(3),
    }

    comparision_table = {
        (0,0,0,0) : 0,
        (0,0,0,1) : 1,
        (0,0,1,0) : 2,
        (0,0,1,1) : 3,
        (0,1,0,0) : 4,
        (0,1,0,1) : 5,
        (0,1,1,0) : 6,
        (0,1,1,1) : 7,
        (1,0,0,0) : 8,
        (1,0,0,1) : 9,
        (1,0,1,0) : 10,
        (1,0,1,1) : 11,
        (1,1,0,0) : 12,
        (1,1,0,1) : 13,
        (1,1,1,0) : 14,
        (1,1,1,1) : 15
    }
    final_table = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 6,
        6: 8,
        7: 9,
        8: 10,
        9: 11,
        10: 12,
        11: 14,
        12: 5,
        13: 7,
        14: 13,
        15: 15
    }
    M= int(mval)
    Bits_per_symbol=np.log2(M)
    Order=pow(2,Bits_per_symbol)
    
    bit_arr = np.array([int(ite) for ite in bitin])


    lx= len(bit_arr)
    m=int(np.log2(M))
    nt= (lx + m - 1)//m
    b= -lx + nt*m

    input_bit_arr = np.pad(bit_arr,(0,b),'constant')
    no_bits = np.size(input_bit_arr)
    bit_period = pow(10,-6)
    symbol_period = bit_period * Bits_per_symbol
    symbol_rate = 1//symbol_period
    FreqC = 2 * symbol_rate
    t = np.arange(symbol_period/100, symbol_period + symbol_period/100, symbol_period/100)
    ss = len(t)
    #print(ss)


    bit = np.empty(0)

    for i in range(len(input_bit_arr)):
        if  input_bit_arr[i] == 1:
            seq = np.ones(100, dtype=int)
        else:
            seq = np.zeros(100, dtype=int)
        bit = np.append(bit,seq)


    t1 = np.arange(bit_period/100, (100*bit_period*len(input_bit_arr))/100, bit_period/100)
    t1 = np.append(t1,16*bit_period/100)
    #print(len(t))

    


    data_reshape = np.reshape(input_bit_arr, (int(no_bits/np.log2(M)),int(np.log2(M))))
    #print(data_reshape)

    l = (int(np.log2(M)),1)
    value = np.zeros(l,dtype=int)
    #print(value)
    value_temp = 0

    for i in range(int(np.log2(M))):
        value_temp = 0
        for j in range(int(no_bits/np.log2(M))):
           # print(data_reshape[i,j])
           value_temp = value_temp + pow(2 ,(int(no_bits/np.log2(M))-j-1)) * data_reshape[i,j]
            #print(pow(2 ,(int(no_bits/np.log2(M))-j-1)) * data_reshape[i,j])
        value[i,0] = value_temp

        Real_part = np.zeros(l,dtype=int)
    Imaginary_part = np.zeros(l,dtype=int)



    for i in range(int(np.log2(M))):
        for b3 in [0, 1]:
            for b2 in [0, 1]:
                for b1 in [0, 1]:
                    for b0 in [0, 1]:
                        B = (b3, b2, b1, b0)
                        C   = (b3, b2, b1, b0)
                        #print(B)
                        Q = mapping_table[B]
                        Q1 = comparision_table[C]
                        fig2 = plt.figure(2)
                        plt.plot(Q.real, Q.imag, 'r-')
                        plt.text(Q.real, Q.imag + 0.2, "".join(str(x) for x in B), ha='center')
                        if value[i,0] == Q1:
                            Real_part[i,0] = Q.real
                            Imaginary_part[i,0] = Q.imag
                            plt.plot(Real_part[i,0], Imaginary_part[i,0], 'bo')
                            plt.text(Real_part[i,0], Imaginary_part[i,0] + 0.2, "".join(str(x) for x in B), ha='center')
                            fig2.savefig('constellation.png')
    plt.close()
                            


    QAM_signal = np.empty(0)

    for i in range(int(np.log2(M))):
        out_real = Real_part[i,0] *  np.cos(2 * np.pi * FreqC * t)
        out_img = Imaginary_part[i,0] * np.sin(2 * np.pi * FreqC * t)
        out = out_img + out_real
        QAM_signal = np.append(QAM_signal,out)

    #print(QAM_signal)
    #print(np.size(QAM_signal))
    tt = np.arange(symbol_period/100, symbol_period*len(Real_part) + symbol_period/100, symbol_period/100)
    #tt = np.append(tt,1)

    


    def calc_fft(y):
        n = len(y)  # length of the signal
        k = np.arange(n)
        t2 = symbol_period
        frq = k / t2  # two sides frequency range
        frq = frq[range(n // 2)]  # one side frequency range
        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(n // 2)]
        return Y, frq

    Z, frq = calc_fft(QAM_signal)

    # Demodulation
    bb = np.split (QAM_signal,int(no_bits/np.log2(M)))

    amp = []
    for i in bb:
        amp.append(int(max(i)))

    print(np.transpose(amp))
    amplitude=np.transpose(amp)
    demo_check=np.array([])

    for k in range(1,13,1):
            phi= phase_table[k]
            y=np.sin((2 * np.pi * FreqC * t) + phi)
            demo_check=np.append(demo_check,y)
    c= np.split(demo_check,12)

    print(len(c[0]))
    d=0
    demo=[]
    for i in bb:
        op=np.array([])
        #d=0
        tp=np.array([])
        tp=np.append(tp,bb[d])
        #for j in c:
            #tp=np.array([])
            #tp=np.append(tp,c[d])
        key=np.dot(c,tp)
        #print(key)
        #print(len(tp))
        op=np.append(op,key)
        d=d+1
        #print(d)
        demo.append(np.argmax(op))
        #print(np.argmax(op))
        #print(demo)

    pp = np.ones(int(no_bits/np.log2(M)), dtype=int)
    final_index = np.transpose(pp) + demo

    #print(final_index)

    received_bit=np.array([])
    for i in range(int(no_bits/np.log2(M))):

        if final_index[i] == 1 :
            if amplitude[i] < 2:
                final_index[i]=12
        if final_index[i] == 3:
            if amplitude[i] < 2:
                final_index[i]=13
        if final_index[i] == 7:
            if amplitude[i] < 2:
                final_index[i]=14
        if final_index[i] == 9:
            if amplitude[i] < 2:
                final_index[i]=15
        key_list = list(comparision_table.keys())
        value_list = list(comparision_table.values())
        received_bit= np.append(received_bit,key_list[value_list.index(final_table[final_index[i]])])
    print("---",final_index)
    #key_list=list(comparision_table.keys())
    #value_list=list(comparision_table.values())
    #print(key_list[value_list.index(final_table[final_index[0]])])
    #print(received_bit)

    demodulated_bits = []
    for i in range(len(received_bit)):
        if  received_bit[i] == 1:
            seq = np.ones(100, dtype=int)
        else:
            seq = np.zeros(100, dtype=int)
        demodulated_bits = np.append(demodulated_bits,seq)


    t4 = np.arange(bit_period/100, (100*bit_period*len(input_bit_arr))/100 + bit_period/100, bit_period/100)
    #t4 = np.append(t1,16*bit_period/100)
    #print(len(t))
    


    fig1 = plt.figure(1)
    plt.plot(t1, bit, 'b+')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title( 'a digital signal' )
    fig1.savefig('input.png')
    plt.close()

    fig3 = plt.figure(3)
    plt.plot(tt,QAM_signal,'bo')
    plt.savefig('modulated.png')
    plt.close()

    fig4 = plt.figure(4)
    plt.plot(frq, abs(Z), 'r')
    plt.savefig('fft.png')
    plt.close()
    
    fig5 = plt.figure(5)
    plt.plot(t4, demodulated_bits, 'r+')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title( 'a digital received_signal')
    fig5.savefig('demodulated.png')
    plt.close()

      

if __name__ == '__main__':
    MyApp().run()
        
