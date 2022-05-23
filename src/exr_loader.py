import OpenEXR
import Imath
import numpy as np
import struct

# the data return is like opencv
# left top is (0,0)
def EXRLoaderRGB(dataPath):
    '''
    Args:
        dataPath: the exr data path
        fg: is the front object
    Returns:
        if fg, return [r, g, b, a, s]
        else, return [a, g, b]
    '''
    File = OpenEXR.InputFile(dataPath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in "RGB"]
    r =np.reshape(rgb[0],(Size[1],Size[0]))
    g =np.reshape(rgb[1],(Size[1],Size[0]))
    b =np.reshape(rgb[2],(Size[1],Size[0]))
    return [r, g, b]

def EXRLoaderRGBX(dataPath,name_list):
    File = OpenEXR.InputFile(dataPath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in name_list]
    return [np.reshape(rgb[ii],(Size[1],Size[0])) for ii in range(len(name_list))]


# add for complex input
# input should be a dictory of str & numpy array
def EXRWriterMega(savePath, name_data,width,height):
    '''
    def channel_to_name(d):
        c_list=['a','b','c','d','e','f','g','h','i','j']
        name = ""
        s = "%03d" % d
        for ii in s:
            name = name + c_list[int(ii)]
        return name
    '''
    header = OpenEXR.Header(width, height)
    d = {}
    ch = {}
    for item in name_data.items():
        name = item[0]
        data = item[1]
        dtype_float16 = np.issubdtype(data.dtype,np.float16)
        dtype_float32 = np.issubdtype(data.dtype,np.float32)
        assert(dtype_float16 or dtype_float32)
        assert(len(data.shape) == 3)
        assert(data.shape[1] == height and data.shape[2] == width)
        for i_channel in range(data.shape[0]):
            to_save = data[i_channel].flatten()
            if data.shape[0] == 3:
                _3name = ['.R','.G','.B']
                channel_name = name + _3name[i_channel]
            elif data.shape[0] == 2:
                _2name = ['.X','.Y']
                channel_name = name + _2name[i_channel]
            elif data.shape[0] == 1:
                channel_name = name + '.S'
            else:
                #channel_name = '%s.%03d' % (name,i_channel)
                #channel_name = name + '.' + channel_to_name(i_channel)
                channel_name = name + '%3d.S' % i_channel
            ch[channel_name] = Imath.Channel(Imath.PixelType(OpenEXR.HALF if dtype_float16 else OpenEXR.FLOAT))
            d[channel_name] = to_save
    header['channels'] = ch
    exrWriter = OpenEXR.OutputFile(savePath, header)
    exrWriter.writePixels(d)
    exrWriter.close()

def EXRWriter(dataPath, width, height, data, channel = 3):
    '''
    Args:
        dataPath: the exr file save path
        width: image width
        height: image height
        data: image data
        fg: is the front object
    '''
    if channel == 3:
        d = {}
        d['R'] = data[0]
        d['G'] = data[1]
        d['B'] = data[2]
        exrWriter = OpenEXR.OutputFile(dataPath, OpenEXR.Header(width, height))
        exrWriter.writePixels(d)
    elif channel == 4:
        header = OpenEXR.Header(width, height)
        d = {}
        ch = {}
        ch['R'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
        ch['G'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
        ch['B'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
        ch['A'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
        d['R'] = data[0]
        d['G'] = data[1]
        d['B'] = data[2]
        d['A'] = data[3]
        header['channels'] = ch
        exrWriter = OpenEXR.OutputFile(dataPath, header)
        exrWriter.writePixels(d)
    else:
        header = OpenEXR.Header(width, height)
        ch = {}
        d = {}
        for i in range(channel):
            ch['P' + str(i)] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
            d['P' + str(i)] = data[i]
        header['channels'] = ch
        exrWriter = OpenEXR.OutputFile(dataPath, header)
        exrWriter.writePixels(d)
    exrWriter.close()

def getrawfloat(filename):
    x = np.fromfile(filename, dtype='float32', count=-1)
    x = x.reshape(4,4)
    x = np.transpose(x)
    #print(x)
    return x

def dumprawfloat(filename,data):
    idata = np.asarray(np.transpose(data),dtype='float32')
    idata.tofile(filename)

def getcameralist(filename):
    f = open(filename,'rb')
    data = f.read(4)
    n = struct.unpack("i", data)[0] # read int
    camera_list=[]
    for ii in range(n):
        data = f.read(16 * 4)
        m = struct.unpack("16f",data) # write 16 float
        m = np.array(m,dtype="float32")
        m = np.reshape(m,[4,4])
        m = np.transpose(m)
        camera_list.append(m)
    f.close()
    return np.array(camera_list)

def dumpcameralist(filename,data):
    f = open(filename,'wb')
    n = len(data)
    f.write(struct.pack("i",n))
    for ii in range(n):
        idata = np.transpose(data[ii])
        idata = idata.flatten().tolist()
        m = struct.pack("16f",*idata)
        f.write(m)
    f.close()

def getposlist(filename):
    f = open(filename,'rb')
    data = f.read(4)
    n = struct.unpack("i", data)[0] # read int
    pos_list=[]
    for ii in range(n):
        data = f.read(3 * 4)
        m = struct.unpack("3f",data) # write 3 float
        m = np.array(m,dtype="float32")
        pos_list.append(m)
    f.close()
    return np.array(pos_list)

def dumpposlist(filename,data):
    f = open(filename,'wb')
    n = len(data)
    f.write(struct.pack("i",n))
    for ii in range(n):
        idata = data.flatten().tolist()
        m = struct.pack("3f",*idata)
        f.write(m)
    f.close()

