
from history import save_detection, read_history

save_detection("Test", 5, 2)

history = read_history()

print(history)
