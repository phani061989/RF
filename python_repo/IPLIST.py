def get_IPList():
    import pysftp

    # Get current file from bigshare
    myHostname = "netsftp.uibk.ac.at"
    myUsername = "x2241036"
    myPassword = "Ugu9Pap3"

    # Connect to sftp server and update IP_Dict
    try:
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        with pysftp.Connection(host=myHostname,
                               username=myUsername,
                               password=myPassword,
                               cnopts=cnopts) as sftp:
            # Switch to Devices directory and download current IPList
            with sftp.cd('/mnt/na04/share04/quantumcircuits/Lab/Devices'):
                sftp.get('IP_Dict.py')
    except:
        # If no connection is possible, just use local IPList
        pass

    # Read IP list
    with open('IP_Dict.py', 'r') as f:
        IPList = eval(f.read())
    
    return IPList
    
    
IPList = get_IPList()