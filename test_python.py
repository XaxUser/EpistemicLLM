fact = "green_pepper_at_blue_crate"
obj, location = fact.split("_at_")
print(f"Object: {obj}, Location: {location}")

obj = "green_pepper"
fact = "green_pepper_at_blue_crate"
final_location = fact.split(f"{obj}_at_")[1]
print(final_location)  # doit afficher: blue_crate