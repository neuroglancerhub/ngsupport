def normalize_server(f):
    def wrapped(server, *args, **kwargs):
        if server.startswith('http-'):
            server = server.replace('http-', 'http://')
        elif server.startswith('https-'):
            server = server.replace('https-', 'https://')
        elif not server.startswith('http'):
            # check for an explicit port, e.g. 'emdata4:8900'
            if ':' in server:
                server = f'http://{server}'
            else:
                server = f'https://{server}'

        return f(server, *args, **kwargs)
    return wrapped
