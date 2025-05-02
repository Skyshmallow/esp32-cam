from newLook import newLook

if __name__ == "__main__":
    # Start with person tracking enabled and recording disabled
    # You can change these parameters as needed
    newLook(follow_person=True, record=True)
    
    # Additional examples (comment out the line above and uncomment one of these to use):
    
    # Just record without tracking
    # newLook(follow_person=False, record=True)
    
    # Both tracking and recording
    # newLook(follow_person=True, record=True)
    
    # Hide debug window (headless mode)
    # newLook(follow_person=True, record=True, show_debug=False)