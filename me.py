from pymodbus.client import ModbusTcpClient
client = ModbusTcpClient(host= '192.168.1.11' , port = 502 )
if client.connect():
    print("Connected to Modbus server")
else:
    print("Failed to connect to Modbus server")
    exit()
# HR 40001 IS FOR REJECTION  
# HR 40002 IS FOR VFD SPEED
# client.write_register(address,value 0 for accepted  , 1 for Rejected ) 
client.write_register(0,0)
#  client.write_register(address,value for VFD Speed  ) 
#client.write_register(1,50)
