import DGuru

while True:
    text = input("DGuru > ")
    result, error = DGuru.run('<stdin>', text)
    
    if error: print(error.as_string())
    else: print(result)