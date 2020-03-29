#Definir funciones

def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z

print(add_numbers(1,2))
print(add_numbers(1,2,3))

def add_numebers1(x,y,z=None,Flag=False):
    if  (Flag==True):
        print("Flag is true")
    if  (z==None):
        x+y
    else:
        x+y+z

print(add_numebers1(1,2,3,True))

a=add_numbers

print(a(1,2))


#Types and Sequences

type('This is a string')
type(1.0)
print(type(1.))

#listas

x=(1,'a',2,'b',3)
print(x)
x.append(3.3)
print(x)