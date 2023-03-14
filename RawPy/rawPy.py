'''
list of all the functions defined so far for analyzing raw data out of the biaxial apparatus BRAVA
(plotting tools are at the end)
'''

### handling file input / output ###

### handling file input / output ###
def load_tdms(exp_name):
    '''
    This function load into a pd.dataframe the data from the biassiale TDMS file.
    it prints out all the metadata regarding the experiments.
    the script should be contained in the same directory as the raw data of the experiment
    input: 
        - file name
    Output:
        - on screen printed information 
        - dataframe containing all the data and comments
    History:
        - created on 10/28/22
    '''
    from nptdms import TdmsFile
    import numpy as np
    import pandas as pd

    try:
        f = open(exp_name, 'rb')
    except:
        print(f'Error Opening {exp_name}')
        return 0
    
    tdms_file = TdmsFile('%s'%(exp_name))                          # open the tdms file 
    metadata, ADC = tdms_file.groups()                             # unpack the groups this is necessary to read the calibration numbers from ADC group

    ###############

    df = tdms_file.as_dataframe()                                  # import tdms into a dataframe

    # extract calibration numbers for each channel
    names=[]
    jj=0
    for i in df.columns:                                           # use dataframe to then get the name of the recorded channels
        if jj<=1:                                                  # this is to skip the first two columns that are not in the ADC group and are comment_time and comment
            pass
            jj+=1
        else:
            names.append(i.split('/')[-1][1:-1])                   # creates a list of column names


    #######################
    new_names=[]                                                   # fix the columns names usable as dataframe index (i.e _ and not space)
    for i in df.columns:
        new_names.append(i.split('/')[-1][1:-1].replace(' ','_'))
    df = df.set_axis(new_names, axis='columns')

    ## make changes to channels ## 
    time_zero = df.Time[0]
    df.Time = df.Time-df.Time[0]                                   # start time at zero

    df['Rec_n'] = np.arange(0,len(df.Time))                        # create a column for the rec_n

    df['NaN'] = np.NaN

    new_names = new_names + ['Rec_n']                              # add it to the list name to show on plot below 
    new_names = new_names + ['NaN']                                # add a column of NaN as option for the plotting below 

    df.Comment_Time = df.Comment_Time.astype(float) - time_zero    # correct the cooment time to start at zero time of the exp

    ###############
    ## use the calibration numbers from the TDMS file to convert volt to engineering units ## 

    ###############
    ## use the calibration numbers from the TDMS file to convert volt to engineering units ## 

    for i,j in zip(names,new_names[2:-2]):                              # loop into the two name lists, the new names are indicized to avoid the comments and the last two cols (Rec_n and NaN)
        foo = list(ADC['%s'%i].properties.items())                      # for each channel create a list of slope and intercept in foo
        df['%s'%j] = df['%s'%j]*foo[0][1]+foo[1][1]                     # use slope and intercept to make the conversion 

    ### print infos as output ###
    print('----------------------------')
    print('--- experimental details ---')
    print('')
    for name, value in tdms_file.properties.items():
        print("{0}: {1}".format(name, value))
    print('')
    print('----------------------------')
    print('')
    print (df.Comment[0])
    print('')
    print('--------Calibrations-----------')
    print ('')
    for i in names: # for each channel extract information on slope and intercept
            for name, value in ADC['%s'%i].properties.items():
                foo = "{0} = {1}".format(name, value)
                print(i, foo)
            print('')
    print('')

    ### comments ###
    time_comm_obj, comm = metadata.channels()
    for i in np.arange(0,len(time_comm_obj)):
        print('')
        print('Time: ',float(time_comm_obj[i])-time_zero)
        print(comm[i])
    return df

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
        return 0

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

def _binary_tuple_to_string(binary_form):
    binary_form = [c.decode() for c in binary_form]
    return ''.join(binary_form)
    
def read_binary(filename, dataendianness='little', pandas=False):
    import numpy as np
    import struct
    import pandas as pd
    """
    Takes a filename containing the binary output from xlook and
    reads the columns into a rec array or dataframe object for easy
    data processing and access.
    The data section of the file is written in the native format of the machine
    used to produce the file.  Endianness of data is little by default, but may
    be changed to 'big' to accomodate older files or files written on power pc
    chips.
    """

    try:
        f = open(filename, 'rb')
    except:
        print(f'Error Opening {filename}')
        return 0

    col_headings = []
    col_recs = []
    col_units = []

    # Unpack information at the top of the file about the experiment
    name = struct.unpack('20c', f.read(20))
    name = _binary_tuple_to_string(name)
    name = name.split('\0')[0]
    print(f'\nName: {name}')

    # The rest of the header information is written in big endian format

    # Number of records (int)
    num_recs = struct.unpack('>i', f.read(4))
    num_recs = int(num_recs[0])
    print(f'Number of records: {num_recs}')

    # Number of columns (int)
    num_cols = struct.unpack('>i', f.read(4))
    num_cols = int(num_cols[0])
    print(f'Number of columns: {num_cols}')

    # Sweep (int) - No longer used
    swp = struct.unpack('>i', f.read(4))[0]
    print(f'Swp: {swp}')

    # Date/time(int) - No longer used
    dtime = struct.unpack('>i', f.read(4))[0]
    print(f'dtime: {dtime}')

    # For each possible column (32 maximum columns) unpack its header
    # information and store it.  Only store column headers of columns
    # that contain data.  Use termination at first NUL.
    for i in range(32):

        # Channel name (13 characters)
        chname = struct.unpack('13c', f.read(13))
        chname = _binary_tuple_to_string(chname)
        chname = chname.split('\0')[0]

        # Channel units (13 characters)
        chunits = struct.unpack('13c', f.read(13))
        chunits = _binary_tuple_to_string(chunits)
        chunits = chunits.split('\0')[0]

        # This field is now unused, so we just read past it (int)
        gain = struct.unpack('>i', f.read(4))

        # This field is now unused, so we just read past it (50 characters)
        comment = struct.unpack('50c', f.read(50))
        comment = _binary_tuple_to_string(comment)

        # Number of elements (int)
        nelem = struct.unpack('>i', f.read(4))
        nelem = int(nelem[0])

        if chname[0:6] == 'no_val':
            continue  # Skip Blank Channels
        else:
            col_headings.append(chname)
            col_recs.append(nelem)
            col_units.append(chunits)

    # Show column units and headings
    print('\n\n-------------------------------------------------')
    print('|%15s|%15s|%15s|' % ('Name', 'Unit', 'Records'))
    print('-------------------------------------------------')
    for column in zip(col_headings, col_units, col_recs):
        print('|%15s|%15s|%15s|' % (column[0], column[1], column[2]))
    print('-------------------------------------------------')

    # Read the data into a numpy recarray
    dtype = []
    for name in col_headings:
        dtype.append((name, 'double'))
    dtype = np.dtype(dtype)

    data = np.zeros([num_recs, num_cols])

    for col in range(num_cols):
        for row in range(col_recs[col]):
            if dataendianness == 'little':
                data[row, col] = struct.unpack('<d', f.read(8))[0]
            elif dataendianness == 'big':
                data[row, col] = struct.unpack('>d', f.read(8))[0]
            else:
                print('Data endian setting invalid, please check and retry')
                return 0

    data_rec = np.rec.array(data, dtype=dtype)

    f.close()

    if pandas:
        # If a pandas object is requested, make a data frame
        # indexed on row number and return it
        dfo = pd.DataFrame(data, columns=col_headings)
        # Binary didn't give us a row number, so we just let
        # pandas do that and name the index column
        dfo.index.name = 'row_num'
        return dfo

    else:
        # Otherwise return the default (Numpy Recarray)
        data_rec = np.rec.array(data, dtype=dtype)
        return data_rec

def save_data(exp_name, var, callingLocals=locals()):
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
        items[:,kk] = i
        kk+=1
    
    # output file 
    # comma separated for easy import with both pandas or numpy
    exp_name = exp_name.split('/')
    exp_name = exp_name[-1]
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
    return col


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
    return col


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
    from pandas.core.series import Series
    
    # Type conversion from Pandas to NumPy
    if type(stress) == type(Series()):
        stress = stress.values
    if type(disp) == type(Series()):
        disp = disp.values

    stiffness = k #[MPa/mm]
    
    # Increments in elastic distortion
    dload = (stress[1:] - stress[:-1]) / stiffness
    # Increments in total displacement
    ddisp = disp[1:] - disp[:-1]
    # Subtract elastic distortion from total displacement
    ec_disp = np.hstack([0, np.cumsum(ddisp - dload)])
    return ec_disp


def shear_strain(ec_disp,lt):
    import numpy as np
    from pandas.core.series import Series
    '''
    Input:
        vertical ec disp
        layer thickness
    Return:
        engineering shear strain
    '''
    if type(ec_disp) == type(Series()):
        ec_disp = ec_disp.value
    if type(lt) == type(Series()):
        lt = lt.value

    strain = np.zeros((len(lt)))

    for i in np.arange(1,len(strain)):
        if i < len(strain):
            strain[i] = strain[i-1]+(ec_disp[i]-ec_disp[i-1]) / ((lt[i]+lt[i-1])/2.0)
    return strain


def filter_low_pass(x, filt_data, cutoff, fs, order=1, rows=[0,-1]):
    """
    Perform a low_pass filter with two poles
    on a 1-d array and the return the filtered signal. 

    Input :
        x       coordinates to produce the comparison plot 
        data    to be filtered
        cutoff  cutoff frequency [Hz]
        fs      1/dt corresponding to the sample frequency in Hz
        order:  number of the corners (default=1)
        You can select a portion of the experiment by passing the argument rows=[row1,row2]
        default is all experiment

    Return:
        array of filtered data 
    
    """

    from scipy.signal import butter, filtfilt
    import numpy as np
    import matplotlib.pyplot as plt 

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a 

    b, a = butter_lowpass(cutoff, fs, order=order)
    
    slice_exp = filt_data[rows[0]:rows[1]]
    lp_slice_exp = filtfilt(b, a, slice_exp)
    lp = np.append(filt_data[:rows[0]],lp_slice_exp)
    lp = np.append(lp,filt_data[rows[1]:])

    f, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(x,filt_data,'k',label='original data')
    ax.plot(x,lp,'r',label='filtered data')
    ax.legend(loc='best')

    return lp


def slope(x, y, row_s, row_e, degree=1):
    '''
    Polynominal fit to experimental data
    
    Input:
        x array
        y array
        row start
        row end 
        degree of fitting (default 1 (linear))
    Return:
        Coefficients of the fitting curve (from the lower to higher degree)
    '''


    import numpy.polynomial.polynomial as poly
    import numpy as np 
    import matplotlib.pyplot as plt 
    
    if degree==1:
        coefs = poly.polyfit(x[row_s:row_e], y[row_s:row_e], 1)
        r_squared = 1 - (sum((y[row_s:row_e] - (coefs[1] * x[row_s:row_e] + coefs[0]))**2) / ((len(y[row_s:row_e]) - 1) * np.var(y[row_s:row_e], ddof=1)))
        x_new = np.linspace(x[row_s], x[row_e], num=len(x)*10)
        ffit = poly.polyval(x_new, coefs)

        f, ax = plt.subplots(1,1,figsize=(8,8))
        ax.plot(x[row_s:row_e],y[row_s:row_e],'k',label='Data')
        ax.plot(x_new,ffit,'r',label=r'Fit')
        ax.set_title(r'slope=%f intercept=%f $R^2$=%f'%(coefs[1],coefs[0],r_squared),color='r') 
        ax.legend(loc='best')
    else:
        coefs = poly.polyfit(x[row_s:row_e], y[row_s:row_e], degree)
        x_new = np.linspace(x[row_s], x[row_e], num=len(x)*10)
        ffit = poly.polyval(x_new, coefs)
        f, ax = plt.subplots(1,1,figsize=(8,8))
        ax.plot(x[row_s:row_e],y[row_s:row_e],'k',label='data')
        ax.plot(x_new,ffit,'r',label='Fit')
        ax.legend(loc='best')
    
    return coefs


def rslope(x,y,window):
    """
    Takes a data vector and a window to produce a vector of the running average slope.
    The window specifies the number of points on either side of the central point, so
    the total number of points in the slope fitting is 2*window+1.  Fitting is 
    done by the least squares method where the slope is defined by the equation below.
    the beginning and ends are padded with NaN, so fewer points are in those slope 
    estimates.  Addition and subtraction to the totals is used so that the sum is not
    recomputed each time, speeding the process.
    
                    sum(x)*sum(y)
        Sum(x*y) -  -------------
                          n
    m = -------------------------
                     (sum(x))^2
        sum(x^2) - --------------
                          n
    """
    
    import numpy as np
    
    # Check that x and y are the same length
    if len(x) != len(y): 
        print ("Error: x and y must be the same length")
        return 0     
    
    N = len(x) # Number of points in the dataset
    slopes = np.ones(N) # Make array for slopes
    
    # Pad data with window number of points NaN on either side
    x_padded = np.empty(2*window+N)
    x_padded[0:window] = 0
    x_padded[window:N+window] = x
    x_padded[N+window:2*window+N] = 0
    
    y_padded = np.empty(2*window+N)
    y_padded[0:window] = 0
    y_padded[window:N+window] = y
    y_padded[N+window:2*window+N] = 0
    
    sum_x    = np.sum(x_padded[0:2*window+1])
    sum_y    = np.sum(y_padded[0:2*window+1])
    sum_x_sq = np.sum(x_padded[0:2*window+1]*x_padded[0:2*window+1])
    sum_xy   = np.sum(x_padded[0:2*window+1]*y_padded[0:2*window+1])

    n = np.empty(N)
    n[0:window] = np.arange(window+1,2*window+1)
    n[window:N-window] = window*2+1
    n[N-window:N] = np.arange(2*window,window,-1)
    
    slopes[0] = (sum_xy - (sum_x*sum_y/n[0]))/(sum_x_sq - (sum_x*sum_x/n[0]))
    
    for i in range(1,N):
        sum_x    = sum_x - x_padded[i-1] + x_padded[2*window+i]
        sum_y    = sum_y - y_padded[i-1] + y_padded[2*window+i]
        sum_x_sq = sum_x_sq - x_padded[i-1]*x_padded[i-1] + \
            x_padded[2*window+i]*x_padded[2*window+i]
        sum_xy   = sum_xy - x_padded[i-1]*y_padded[i-1] +\
            x_padded[2*window+i]*y_padded[2*window+i]
        slopes[i] = (sum_xy - (sum_x*sum_y/n[i]))/(sum_x_sq - (sum_x*sum_x/n[i]))
    return slopes


def rgt(disp,lt,L=50):
    import numpy as np 
    '''
    This function applies a correction for the geometrical thinning of the sample in double direct shear.
    It is based on the paper Scott et al. 1994 JGR (tringular removal)

    Input:
        - Load point displacement
        - Layer thickness 
        - Length of the forcing block (default 50[mm])
    Return:
        - array of corrected values of layer thickness
    
    '''
    ##
    from pandas.core.series import Series
    
    # Type conversion from Pandas to NumPy
    if type(disp) == type(Series()):
        disp = disp.values
    if type(lt) == type(Series()):
        lt = lt.values

    ## triangular model
    rgt = np.zeros((len(lt)))
    rgt[0]=lt[0]
    dh = 0
    for i in np.arange(1,len(lt)):
        if i < len(lt):
            dh = dh + lt[i-1]*(disp[i]-disp[i-1])/(2*L)
            rgt[i] = lt[i-1]+dh
    return rgt    

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
