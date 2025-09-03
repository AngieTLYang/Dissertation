To let the server accept connections on the required ports, you need to allow inbound TCP connections. Run the following commands in an Administrator Command Prompt:

```cmd
netsh advfirewall firewall add rule name="Allow Command Port 12345" dir=in action=allow protocol=TCP localport=12345
netsh advfirewall firewall add rule name="Allow Command Port 12346" dir=in action=allow protocol=TCP localport=12346
