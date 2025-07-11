import speech_recognition as sr

def main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
         r.adjust_for_ambient_noise(source, duration=1)
         print("Please speak something:")
         audio = r.listen(source)
         print("Processing audio....")
         sst = r.recognize_google(audio)
         print("You said: " + sst)
           
           
          

main()
