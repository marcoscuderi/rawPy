'''
list of all the functions defined so far for analyzing raw data out of the biaxial apparatus BRAVA
(plotting tools are at the end)
'''

### handling file input / output ###


def load_data(filename,pandas=False):
    import numpy as np
    import pandas as pd
    '''
    This function load the raw data produced by the BRAVA .vi
    It should be used to start the process of reducing raw_data 
    created 7/4/2020

    arguments input: 
        path to the data file
        
    return: 
        a rec_array of the data
        if pandas = True it returns a pandas DataFrame
    TO DO: 
        allow for pandas (?) - maybe not necessary at this stage of the analysis 
    '''
    # open file
    try: 
        f=open(filename,'r')
    except:
        print ('error opening %s'%filename)
        #return 0

    # set up col headings
    col_heading = f.readline()
    col_heading = col_heading.split('\t')
    col_heading[-1] = col_heading[-1][:-1] #last col is rec# 

    # set up units 
    col_units = f.readline()
    col_units = col_units.split('\t')
    col_units[-1] = col_units[-1][:-1] #last col is rec# 

    f.close() #close the file 

    # set the headings for the rec array
    dtype = []
    for name in col_heading:
        dtype.append((name,'float'))
    dtype=np.dtype(dtype)
    # load the data in a matrix
    data = np.loadtxt(filename,skiprows=2)
    # get the number of records
    records = int(data[-1,-1])
    
    print ('------------------------------------------------------')
    print ('|%20s|%15s|%15s|'%('Name','Unit','Records'))
    print ('-----------------------------------------------------')
    for column in zip(col_heading,col_units):
        print ('|%20s|%15s|%15s|'%(column[0],column[1],records))
    print ('------------------------------------------------------')

    if pandas==True:
        pd_frame = pd.DataFrame(data,columns=col_heading)
        return pd_frame
    else:
        # create rec array (add the pandas option)
        rec_array = np.rec.array(data,dtype=dtype)
        
        return rec_array


def save_data(exp_name,var, callingLocals=locals()):
    import numpy as np
    
    names = []
    units =[]
    # get the names of each variable to be saved
    for i in np.arange(len(var)):       
        if i == 0: 
            name = [ k for k,v in callingLocals.items() if v is var[i]]
            name = name[0]
            names.append(name)
        else:
            name = [ k for k,v in callingLocals.items() if v is var[i]]
            names.append(name[0])
    
    # get the units from the name
    for i in names:
        units.append(i.split('_')[-1])
    
    # create a suitable array structure to be saved
    items = np.zeros((len(var[0]),len(var)))
    kk=0
    for i in var:
        items[:,kk] = i.reshape(len(i))
        kk+=1
    
    # output file 
    # comma separated for easy import with both pandas or numpy
    outfile=open('%s_data_rp'%exp_name[:4],'w')
    
    outfile.write('%s\n'%('\t'.join(names)))
    outfile.write('%s\n'%('\t'.join(units)))
    
    np.savetxt(outfile,items,delimiter='\t')
    outfile.close()
    
    print ('Saving data')
    print()
    print ('------------------------------------------------------')
    print ('|%20s|%15s|%15s|'%('Name','Unit','Records'))
    print ('-----------------------------------------------------')
    for column in zip(names,units):
        print ('|%20s|%15s|%15s|'%(column[0],column[1],len(items)))
    print ('------------------------------------------------------')


### Utilities to analize data ###


def zero(col,row):
    '''
    zero the data given the row number
    it authomatically zero the data at the desired row# and remove the noise before it

    inputs: 
        column, row
    returns: 
        array
    '''
    # col = np.append(col[:row]*0,col[row:]-col[row])
    col = col - col[row]
    return col.reshape(len(col),1)


def offset(col,row1,row2):
    import numpy as np
    '''
    Perform offset between two points
    input: 
        column, row1, row2
    return:
        array
    '''
    int_1 = col[:row1]
    int_2 = col[row1:row2]*0+col[row1]
    int_3 = (col[row2:]-col[row2])+col[row1]
    col = np.append(int_1,int_2)
    col = np.append(col,int_3)
    return col.reshape(len(col),1)


def ElasticCorrection(stress,disp,k):
    import numpy as np
    '''
    input:
        stress
        disp
        k = stiffness
    Return: 
        elestic corrected displacement
    '''
    ec_disp=np.zeros((len(disp),1))

    stiffness = k #[MPa/mm]

    for i in np.arange(1,len(disp)):
        if i < len(disp):
            ec_disp[i] = (ec_disp[i-1] + (disp[i]-disp[i-1])) - ((stress[i]-stress[i-1])/stiffness)
    return ec_disp


def shear_strain(ec_disp,lt):
    import numpy as np
    '''
    Input:
        vertical ec disp
        layer thickness
    Return:
        engineering shear strain
    '''
    strain = np.zeros((len(lt),1))

    for i in np.arange(1,len(strain)):
        if i < len(strain):
            strain[i] = strain[i-1]+(ec_disp[i]-ec_disp[i-1]) / ((lt[i]+lt[i-1])/2.0)
    return strain


### plotting ###


class Cursor(object):
    
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.txt = ax.text(0.85, 0.95, '', transform=ax.transAxes,bbox=dict(facecolor='white', edgecolor='red',boxstyle='round,pad=1'))

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        
        self.ax.set_title('x=%d, y=%1.2f' % (x, y))
        self.ax.figure.canvas.draw()
        
    def onclick(self,event):
        if event.button==3:
            x_ev, y_ev = event.xdata, event.ydata
            self.txt.set_text('x=%d \n y=%f' % (x_ev,y_ev))
            self.ax.figure.canvas.draw()


def plot(x,y,n_x='x_label',n_y='y_label'):
    import matplotlib.pyplot as plt
    '''
    Input:
        x,y variable to plot
        x,y axis label
    Return: 
        figure
        cross axis indicate the coordinates of the mouse position
        righ click shows value of x-axis
    
    create a standard function to plot the data during analysis
    it sould have a cursor that indicates the x,y,row_number
    or you can make all the plots in row instead of time (last option since not preatty)
    
    This is a good compromise since it is fast and shows the row number
    Drawback is that the cursor does not highlight each point 
    '''
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x,y, 'k-o')
    ax.set_xlabel('%s'%n_x)
    ax.set_ylabel('%s'%n_y)
    cursor = Cursor(ax)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
    fig.canvas.mpl_connect('button_press_event', cursor.onclick)

    plt.show()
    return cursor
