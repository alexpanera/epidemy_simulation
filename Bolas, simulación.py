# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 01:11:54 2020

@author: Media Service
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial

#para realizar la simulación de movimiento libre, poner 0 en porcentaje_confinados y en porcenataje_cuarentena
#para realizar la simulación de cuarentena, poner 0 en porcentaje_confinados 
#para realizar la simulación de confinamiento, poner 0 en porcentaje_cuarentena 

lado=10 #lado de la caja cuadrada
dt=0.01 #x = v*dt; para actualizar las posiciones 
Npasos=125000 #'tiempo' que dura la simulación
pasosrep = np.arange(0,Npasos,1) #se utiliza después para representar 
pasos=10 #número de pasos entre representación y representación
pausa=0.01 #tiempo real entre representaciones
nhabitantes=200 #número de habitantes de la población
distchoque=0.20 #distancia entre las coordenadas de las bolas para que se considere que están chocando
duracion_enfermedad = 1000 #número de pasos que dura la enfermedad
duracion_inmunidad = 1000  #número de pasos que dura la inmunidad
curados_muerte = 9 #número de habitantes que se curan por cada habitante muerto
porcentaje_confinados = 0.5 #porcentaje de la población que está confinado
rendijax = 2.5 #posición X de la rendija
apertura = 0.04 #longitud de la apertura de la rendija
porcentaje_cuarentena = 0 #porcentaje de la población en cuarentena

def choque(x1,x2,v1,v2): #función utilizada para el choque normal entre dos bolas
    distancia=(((x1-x2)[0])**2+((x1-x2)[1])**2)**0.5
    normal=(x2-x1)/distancia
    tangencial=np.array([-normal[1],normal[0]])
    if np.dot(v1,normal)-np.dot(v2,normal) >= 0:
        v1normal=np.dot(v1,normal)
        v2normal=np.dot(v2,normal)
                
        v1normal,v2normal=v2normal,v1normal
                
        v1tangencial=np.dot(v1,tangencial)
        v2tangencial=np.dot(v2,tangencial)
                
        v1 = normal*v1normal+tangencial*v1tangencial
        v2 = normal*v2normal+tangencial*v2tangencial       
    return v1,v2

def choquequieta(x1,x2,v1): #función para cuando en el choque está involucrada una bola en confinamiento 
    distancia=(((x1-x2)[0])**2+((x1-x2)[1])**2)**0.5
    normal=(x2-x1)/distancia
    tangencial=np.array([-normal[1],normal[0]])
    
    v1normal=np.dot(v1,normal)
    v1normal=-v1normal               
    v1tangencial=np.dot(v1,tangencial) 
              
    v1 = normal*v1normal+tangencial*v1tangencial
    return v1

nrep = 6 #número de veces que lo queremos ejecutar. Es útil si queremos calcular la media de varias ejecuciones

#para representar la media de varias ejecuciones
infect_todos = np.zeros((nrep,Npasos))
sanos_todos = np.zeros((nrep,Npasos))
curados_todos = np.zeros((nrep,Npasos))
muertos_todos = np.zeros((nrep,Npasos))

for n in range(nrep):  
    if porcentaje_cuarentena != 0:
        #coordenadas x
        x_cuarentena = (rendijax-0.2)*np.random.rand(np.int(nhabitantes*porcentaje_cuarentena),1) #restamos 0.2 para que ninguna aparezca justo en la rendija
        x_libres = (lado-rendijax-0.2)*np.random.rand(np.int(nhabitantes*(1-porcentaje_cuarentena)),1) + rendijax + 0.2
        #coordenadas y
        y = lado*np.random.rand(nhabitantes,1)
        
        #posiciones (x,y)
        x = np.zeros((nhabitantes,2))
        x[:np.int(nhabitantes*(1-porcentaje_cuarentena)),0] = x_libres[:,0]
        x[np.int(nhabitantes*(1-porcentaje_cuarentena)):,0] = x_cuarentena[:,0]
        x[:,1] = y[:,0]  
    else:
        x=lado*np.random.rand(nhabitantes,2)
   
    
    #velocidades
    ang0=2*np.pi*np.random.rand(np.int(nhabitantes*(1-porcentaje_confinados))) #dirección inicial de la velocidad de las bolas
    v0=0.5*np.random.rand(2,np.int(nhabitantes*(1-porcentaje_confinados))) #módulo inicial de la velocidad de las bolas
    v_mov=v0.T*np.array([np.cos(ang0),np.sin(ang0)]).T #velocidad inicial de las bolas
    
    v = np.zeros((nhabitantes,2))
    v[np.int(nhabitantes*porcentaje_confinados):] = v_mov
    
    fig=plt.figure(figsize=(20,20)) #representación. Se irá actualizando en cada paso de tiempo
    ax=fig.add_subplot(111)
    plt.title('Cuarentena')
    ##utilizamos una representación para cada grupo de bolas
    if porcentaje_cuarentena != 0:
            ax.axvline(rendijax,0,0.5-apertura/2,color = 'k')
            ax.axvline(rendijax,0.5+apertura/2,1,color = 'k')
    sanos_rep,=ax.plot(x[:-1,0],x[:-1,1],color='limegreen',marker='o',linewidth=0)
    infect_rep,=ax.plot(x[-1,0],x[-1,1],color='indianred',marker='o',linewidth=0)
    curados_rep,=ax.plot(x[3,0],x[3,1],color='cornflowerblue',marker='o',linewidth=0)
    muertos_rep, = ax.plot(x[4,0],x[4,1],color='k',marker='o',linewidth=0)
    ax.set_xlim((0,lado))
    ax.set_ylim((0,lado))

    num_infect = 1 #número de infectados iniciales. Se irá actualizando
    tiempo_infect = np.zeros(nhabitantes) #tiempo que lleva cada bola en el grupo de infectados
    
    num_curados = 0 #número de curados iniciales. Se irá actualizando
    tiempo_curados = np.zeros(nhabitantes) #tiempo que lleva cada bola en el grupo de curados

    retirados_totales = 1 #acumula el número total de habitantes que finalizan la enfermedad, tanto muertos como curados
    
    num_muertos = 0 #número de muertos iniciales. Se irá actualizando
    
    #en estos arrays guardamos el número de habitantes que conforman cada grupo en cada paso para después representarlo
    curadosrep = np.zeros(Npasos)
    infectrep = np.zeros(Npasos)
    sanosrep = np.zeros(Npasos)
    muertosrep = np.zeros(Npasos)
    for i in range(Npasos):
        x=x+dt*v #actualizamos las posiciones 
        
        num_sanos = nhabitantes - num_infect - num_curados - num_muertos

        #definimos los grupos
        sanos = x[:num_sanos,:]
        curados = x[num_sanos: num_sanos+num_curados,:]
        infectados = x[num_curados+num_sanos:nhabitantes-num_muertos,:]
        muertos = x[nhabitantes-num_muertos:,:]        
        
        #a partir de aquí el objetivo es cambiar la velocidad de las bolas cuando chocan 
        #entre ellas o con las paredes y cambiarlas de grupo cuando corresponda
                
        #hallamos los indices de las bolas que están chocando en cada instante
        points_tree=spatial.cKDTree(x)
        pairs=points_tree.query_pairs(distchoque)
        
        #para dichos índices, si ninguna de las bolas pertenece al grupo muertos, les aplicamos la función choque y les cambiamos la velocidad
        for ipair in pairs: #dependiendo de la velocidad de las bolas antes del choque, utilizamos una función u otra
            if ipair[0] < nhabitantes - num_muertos and ipair[1] < nhabitantes - num_muertos:
                if np.all(v[ipair[0],:]) != 0 and np.all(v[ipair[1],:]) != 0:
                    v[ipair[0],:],v[ipair[1],:]=choque(x[ipair[0],:],x[ipair[1],:],v[ipair[0],:],v[ipair[1],:])
                if np.all(v[ipair[0],:]) == 0 and np.all(v[ipair[1],:]) != 0:
                    v[ipair[1],:] = choquequieta(x[ipair[1],:],x[ipair[0],:],v[ipair[1],:])
                if np.all(v[ipair[1],:]) == 0 and np.all(v[ipair[0],:]) != 0:
                    v[ipair[0],:] = choquequieta(x[ipair[0],:],x[ipair[1],:],v[ipair[0],:])
                    
        #cambiamos la velocidad de las bolas que chocan con las paredes
        for j in range(nhabitantes):
            if x[j,0]>=lado:
                if v[j,0] > 0:
                    v[j,0]=-v[j,0]
            if x[j,0]<=0: 
                if v[j,0] < 0:
                    v[j,0]=-v[j,0]
            if x[j,1]>=lado:
                if v[j,1] > 0:
                    v[j,1]=-v[j,1]
            if x[j,1]<=0: 
                if v[j,1] < 0:
                    v[j,1]=-v[j,1]
            if porcentaje_cuarentena != 0:
                if x[j,0] <= (rendijax + 0.10) and x[j,0] >= (rendijax - 0.10):
                    if x[j,1] <= (lado*(0.5-apertura/2)) or x[j,1] >= (lado*(0.5+apertura/2)):
                        v[j,0] = -v[j,0]
                if x[j,0] <= (rendijax + 0.05) and x[j,0] >= (rendijax - 0.05) and x[j,1] >= (lado*(0.5-apertura/2)-0.1) and x[j,1] <= (lado*(0.5-apertura/2)+0.1):
                    v[j,1]=-v[j,1]
                if x[j,0] <= (rendijax + 0.05) and x[j,0] >= (rendijax - 0.05) and x[j,1] >= (lado*(0.5+apertura/2)-0.1) and x[j,1] <= (lado*(0.5+apertura/2)+0.1):
                    v[j,1]=-v[j,1] 
               
        #bucle en el que hacemos que si una bola infectada choca con una sana, ésta pase al grupo de infectados
        for ipair in pairs:
            for k in range(num_sanos+num_curados,nhabitantes-num_muertos): #el indice tiene que pertenecer al grupo de infectados
                if ipair[1] == k and ipair[0] < num_sanos: #una de las bolas tiene que estar infectada y la otra sana (ipair[1]>ipair[0] siempre)
                    infectados = np.insert(infectados,0,np.array([x[ipair[0],:]]),axis=0) #añadimos la bola sana al grupo de infectados
                    sanos = np.delete(sanos,ipair[0],axis=0) #y la quitamos del grupo de sanos
                    
                    pos = np.copy(x[ipair[0],:]) #movemos también las posiciones para que cada bola se siga correspondiendo con su posición
                    x[ipair[0]:num_sanos+num_curados-1,:] = x[ipair[0]+1:num_sanos+num_curados,:]
                    x[num_sanos+num_curados-1,:] = pos
                    
                    vel = np.copy(v[ipair[0],:]) #y lo mismo con las velocidades
                    v[ipair[0]:num_sanos+num_curados-1,:] = v[ipair[0]+1:num_sanos+num_curados,:]
                    v[num_sanos+num_curados-1,:] = vel
                        
                    num_infect += 1 #el grupo de infectados tiene un miembro más
                    num_sanos -=1 #el grupo de sanos tiene un miembro menos
        
        #bucle en el que hacemos que si una bola infectada ha superado el periodo de enfermedad pase al grupo de
        #muertos o de recuperados
        for h in range(nhabitantes):
            if h >= num_curados+num_sanos and h < nhabitantes-num_muertos: #el indice tiene que pertenecer al grupo de infectados
                tiempo_infect[h] += 1 #a cada bola infectada le sumamos 1 paso 
                if tiempo_infect[h] >= duracion_enfermedad: #si el tiempo supera la duracion de la enfermedad
                    if np.mod(retirados_totales,curados_muerte) == 0: #1 de cada 50 veces muere!!!
                        muertos = np.append(muertos,np.array([infectados[h + num_muertos - nhabitantes,:]]),axis=0) 
                        infectados = np.delete(infectados,h + num_muertos - nhabitantes,axis=0)
                        
                        num_infect -= 1 #el grupo de infectados tiene un miembro menos                        
                        num_muertos += 1 #el grupo de muertos tiene un miembro más
                        retirados_totales += 1 #el grupo de retirados totales tiene un miebro más
                    #en este caso no hace falta mover las posiciones y las velocidades, a las bolas muertas simplemente las dejamos
                    #las últimas en el array de posiciones
                    
                    else: #el resto de las veces se cura!!!
                        curados = np.append(curados,np.array([infectados[h + num_muertos - nhabitantes,:]]),axis=0)
                        infectados = np.delete(infectados,h + num_muertos - nhabitantes,axis=0)
    
                        pos = np.copy(x[h - nhabitantes,:])#igual que antes, movemos las posiciones y las velocidades
                        x[num_sanos+num_curados+1:h+1,:] = x[num_sanos+num_curados:h,:]
                        x[num_sanos+num_curados,:] = pos
                                
                        vel = np.copy(v[h - nhabitantes,:])
                        v[num_sanos+num_curados+1:h+1,:] = v[num_sanos+num_curados:h,:]
                        v[num_sanos+num_curados,:] = vel 
                        
                        cop = np.copy(tiempo_infect[h]) #también movemos el tiempo de infectados
                        tiempo_infect[num_sanos+num_curados+1:h+1] = tiempo_infect[num_sanos+num_curados:h]
                        tiempo_infect[num_sanos+num_curados] = 0 #y reseteamos, para que cuando se vuelva a infectar parta de 0
                        
                        num_infect -= 1 #el grupo de infectados tiene un miembro menos
                        num_curados += 1 #el grupo de curados tiene un miembro mas
                        retirados_totales += 1 #el grupo de retirados tiene un miembro mas

        #bucle en el que hacemos que si una bola recuperada ha superado el periodo de inmunidad pase al grupo de sanos      
        for h in range(nhabitantes):
            if h >= num_sanos and h < num_sanos+num_curados: #el indice tiene que pertenecer al grupo de curados 
                tiempo_curados[h] += 1 #sumamos 1 paso de tiempo
                if tiempo_curados[h] >= duracion_inmunidad: #si el tiempo supera la duracion de la inmunidad
                    sanos = np.insert(sanos,0,np.array([curados[0,:]]),axis=0)
                    curados = np.delete(curados,0,axis=0)

                    pos = np.copy(x[h,:]) #movemos las posiciones, las velocidades y los tiempos
                    x[1:h+1,:] = x[:h,:]
                    x[0,:] = pos
                           
                    vel = np.copy(v[h,:])
                    v[1:h+1,:] = v[:h,:]
                    v[0,:] = vel 
                    
                    cop = np.copy(tiempo_curados[h])
                    tiempo_curados[1:h+1] = tiempo_curados[:h]
                    tiempo_curados[0] = 0 #reseteamos
                    
                    num_curados -= 1 #el grupo de curados tiene un miembro menos
                    num_sanos += 1 #el grupo de sanos tiene un miembro mas
        
        #en cada paso de tiempo guardamos el número de miembros de cada grupo
        muertosrep[i] = num_muertos 
        curadosrep[i] = num_curados
        infectrep[i] = num_infect
        sanosrep[i] = num_sanos

        #representamos cada 10 pasos
        if np.mod(i,pasos) == 0:
            sanos_rep.set_data(sanos[:,0],sanos[:,1])
            infect_rep.set_data(infectados[:,0],infectados[:,1])
            #ax.axvline(rendijax,0,0.5-apertura/2,color = 'k')
            #ax.axvline(rendijax,0.5+apertura/2,1,color = 'k')
            curados_rep.set_data(curados[:,0],curados[:,1])
            muertos_rep.set_data(muertos[:,0],muertos[:,1])
            plt.legend(('%s sanos'%np.int(sanosrep[i]),'%s infectados'%np.int(infectrep[i]),'%s curados'%np.int(curadosrep[i]),'%s muertos'%np.int(muertosrep[i])),loc='upper right')
            plt.pause(pausa)

    infect_todos[n] = infectrep #para cada simulación, guardamos la curva de infectados para después representar la media
    curados_todos[n] = curadosrep
    sanos_todos[n] = sanosrep
    muertos_todos[n] = muertosrep
    #representamos la evolución de cada grupo
    plt.figure()
    plt.plot(pasosrep,curadosrep,label='curados',color='b')
    plt.plot(pasosrep,infectrep,label='infectados',color='r')
    plt.plot(pasosrep,sanosrep,label='sanos',color='g')
    plt.plot(pasosrep,muertosrep,label='muertos',color='k')
    plt.legend()
    plt.show()

#representamos la media de los infectados de cada simulación
infect_media = np.mean(infect_todos,axis=0)
curados_media = np.mean(curados_todos,axis=0)
sanos_media = np.mean(sanos_todos,axis=0)
muertos_media = np.mean(muertos_todos,axis=0)

plt.figure()
plt.plot(pasosrep,infect_media,label='infectados',color='r')
plt.plot(pasosrep,curados_media,label='curados',color='b')
plt.plot(pasosrep,sanos_media,label='sanos',color='g')
plt.title('Confinamiento 50%')
plt.xlabel('nº pasos')
plt.ylabel('habitantes')
plt.plot(pasosrep,muertos_media,label='muertos',color='k')
plt.legend()
plt.grid()
plt.show()