import matplotlib.pyplot as plt
import numpy as np
import math
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

M=16
Bits_per_symbol=np.log2(M)
Order=pow(2,Bits_per_symbol)
#print ("Ld=",Ld)
#print ("Lb=",Lb)
if M != Order:
   print("the value of M is only acceptable if log2(M)is an integer")

bit_arr = np.array([0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1])


lx= len(bit_arr)
m=int(np.log2(M))
nt= (lx + m - 1)//m
b= -lx + nt*m
input_bit_arr= np.pad(bit_arr,(0,b),'constant')
no_bits = np.size(input_bit_arr)

print(no_bits)
bit_period = pow(10,-6)
symbol_period = bit_period * Bits_per_symbol
symbol_rate = 1/symbol_period
FreqC = 2 * symbol_rate
print(FreqC)
t = np.arange(symbol_period/100, symbol_period + symbol_period/100, symbol_period/100)
ss = len(t)
print(ss)


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

plt.figure(1)
plt.plot(t1, bit, 'b+')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title( 'a digital signal' )



data_reshape = np.reshape(input_bit_arr, (int(no_bits/np.log2(M)),int(np.log2(M))))
print(data_reshape)

l = (int(no_bits/np.log2(M)),1)
value = np.zeros(l,dtype=int)
#print(value)
value_temp = 0

for i in range(0,int(no_bits/np.log2(M)),1):
    value_temp = 0
    print("==",i)
    for j in range(0,int(np.log2(M)),1):
       # print(data_reshape[i,j])
        print("====",j)
        value_temp = value_temp + pow(2 ,(np.log2(M))-j-1) * data_reshape[i,j]
        #print(pow(2 ,(int(no_bits/np.log2(M))-j-1)) * data_reshape[i,j])
    value[i,0] = value_temp
print(value)

Real_part = np.zeros(l,dtype=int)
Imaginary_part = np.zeros(l,dtype=int)



for i in range(int(no_bits/np.log2(M))):
    for b3 in [0, 1]:
        for b2 in [0, 1]:
            for b1 in [0, 1]:
                for b0 in [0, 1]:
                    B = (b3, b2, b1, b0)
                    C   = (b3, b2, b1, b0)
                    #print(B)
                    Q = mapping_table[B]
                    Q1 = comparision_table[C]
                    plt.figure(2)
                    plt.plot(Q.real, Q.imag, 'r-')
                    plt.text(Q.real, Q.imag + 0.2, "".join(str(x) for x in B), ha='center')
                    if value[i,0] == Q1:
                        Real_part[i,0] = Q.real
                        Imaginary_part[i,0] = Q.imag
                        plt.plot(Real_part[i,0], Imaginary_part[i,0], 'bo')
                        plt.text(Real_part[i,0], Imaginary_part[i,0] + 0.2, "".join(str(x) for x in B), ha='center')

print(Real_part)
print(Imaginary_part)


QAM_signal = np.empty(0)

for i in range(int(no_bits/np.log2(M))):
    out_real = Real_part[i,0] *  np.sin(2 * np.pi * FreqC * t)
    out_img = Imaginary_part[i,0] * np.cos(2 * np.pi * FreqC * t)
    out = out_img + out_real
    QAM_signal = np.append(QAM_signal,out)

#print(QAM_signal)
print(np.size(QAM_signal))
tt = np.arange(symbol_period/100, symbol_period*len(Real_part) + symbol_period/100, symbol_period/100)
#tt = np.append(tt,1)
plt.figure(3)
plt.plot(tt,QAM_signal,'bo')


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
plt.figure(4)
plt.plot(frq, abs(Z), 'r')

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
    print(key)
    print(len(tp))
    op=np.append(op,key)
    d=d+1
    print(d)
    demo.append(np.argmax(op))
    print(np.argmax(op))
    print(demo)

pp = np.ones(int(no_bits/np.log2(M)), dtype=int)
final_index = np.transpose(pp) + demo

print(final_index)

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
print(received_bit)

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
plt.figure(5)
plt.plot(t4, demodulated_bits, 'r+')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title( 'a digital received_signal' )
plt.show()

