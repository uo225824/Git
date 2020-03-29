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

x=[1,'a',2,'b',3]
print(x)
x.append(3.3)
print(x)


for i in x:
    print(i)

i=0
while( i != len(x) ):
    print(x[i])
    i = i + 1


[1,2] + [3,4]
[1]*3
1 in [1, 2, 3]

x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters

x[-4:-2]
x[3:]


firstname = 'Christopher'
lastname = 'Brooks'

print(firstname + ' ' + lastname)
print(firstname*3)
print('Chris' in firstname)


firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)


#Diccionarios

print('DICCIONARIOS')
x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
x['Christopher Brooks'] # Retrieve a value by using the indexing operator

for name in x:
    print(x[name])

for email in x.values():
    print(email)

for name, email in x.items():
    print(name)
    print(email)

x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x

print(fname)