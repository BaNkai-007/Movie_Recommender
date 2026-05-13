import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# MOVIE DATASET
MOVIES_DATA = [
    # (title, year, genres, avg_rating, num_ratings)
    ("Toy Story", 1995, "Animation|Comedy|Family", 4.0, 452),
    ("Jumanji", 1995, "Adventure|Children|Fantasy", 3.2, 180),
    ("GoldenEye", 1995, "Action|Adventure|Thriller", 3.6, 210),
    ("The Matrix", 1999, "Action|Sci-Fi|Thriller", 4.5, 590),
    ("The Dark Knight", 2008, "Action|Crime|Drama|Thriller", 4.6, 650),
    ("Inception", 2010, "Action|Crime|Drama|Mystery|Sci-Fi|Thriller", 4.4, 570),
    ("Batman Begins", 2005, "Action|Crime|Thriller", 4.1, 420),
    ("Batman Returns", 1992, "Action|Crime|Fantasy|Thriller", 3.5, 200),
    ("Batman Forever", 1995, "Action|Adventure|Comedy|Crime", 2.8, 180),
    ("Batman & Robin", 1997, "Action|Adventure|Comedy|Crime|Fantasy", 2.2, 160),
    ("The Avengers", 2012, "Action|Adventure|Sci-Fi|Thriller", 4.3, 530),
    ("Avengers: Endgame", 2019, "Action|Adventure|Drama|Sci-Fi", 4.5, 600),
    ("Iron Man", 2008, "Action|Adventure|Sci-Fi", 4.2, 490),
    ("Thor", 2011, "Action|Adventure|Drama|Fantasy|Sci-Fi", 3.7, 360),
    ("Captain America: The First Avenger", 2011, "Action|Adventure|Sci-Fi|Thriller|War", 3.9, 380),
    ("Spider-Man", 2002, "Action|Adventure|Sci-Fi|Thriller", 3.8, 400),
    ("Spider-Man 2", 2004, "Action|Adventure|Sci-Fi|Thriller", 4.0, 420),
    ("Spider-Man: Into the Spider-Verse", 2018, "Action|Adventure|Animation|Sci-Fi", 4.5, 350),
    ("Interstellar", 2014, "Adventure|Drama|Sci-Fi", 4.4, 540),
    ("The Shawshank Redemption", 1994, "Crime|Drama", 4.6, 700),
    ("Schindler's List", 1993, "Drama|War", 4.5, 580),
    ("Forrest Gump", 1994, "Comedy|Drama|Romance|War", 4.3, 640),
    ("The Silence of the Lambs", 1991, "Crime|Horror|Thriller", 4.3, 520),
    ("Pulp Fiction", 1994, "Crime|Drama|Thriller", 4.5, 610),
    ("Fight Club", 1999, "Drama|Mystery|Thriller", 4.4, 590),
    ("Goodfellas", 1990, "Crime|Drama", 4.4, 500),
    ("The Godfather", 1972, "Crime|Drama", 4.6, 650),
    ("The Godfather Part II", 1974, "Crime|Drama", 4.5, 560),
    ("Scarface", 1983, "Action|Crime|Drama", 4.0, 430),
    ("The Departed", 2006, "Crime|Drama|Thriller", 4.4, 510),
    ("No Country for Old Men", 2007, "Crime|Drama|Thriller", 4.2, 440),
    ("There Will Be Blood", 2007, "Drama|Western", 4.1, 350),
    ("12 Angry Men", 1957, "Drama", 4.5, 430),
    ("One Flew Over the Cuckoo's Nest", 1975, "Drama", 4.4, 490),
    ("Titanic", 1997, "Drama|Romance", 3.9, 610),
    ("The Notebook", 2004, "Drama|Romance", 3.7, 390),
    ("La La Land", 2016, "Drama|Musical|Romance", 4.0, 430),
    ("Casablanca", 1942, "Drama|Romance|War", 4.4, 410),
    ("Romeo + Juliet", 1996, "Drama|Romance", 3.5, 300),
    ("Pride & Prejudice", 2005, "Drama|Romance", 4.0, 360),
    ("Sense and Sensibility", 1995, "Drama|Romance", 3.9, 290),
    ("Bridget Jones's Diary", 2001, "Comedy|Drama|Romance", 3.6, 310),
    ("Sleepless in Seattle", 1993, "Comedy|Drama|Romance", 3.6, 290),
    ("When Harry Met Sally", 1989, "Comedy|Drama|Romance", 4.0, 380),
    ("Crazy, Stupid, Love", 2011, "Comedy|Drama|Romance", 3.9, 350),
    ("The Proposal", 2009, "Comedy|Drama|Romance", 3.5, 300),
    ("Hitch", 2005, "Comedy|Drama|Romance", 3.6, 320),
    ("10 Things I Hate About You", 1999, "Comedy|Drama|Romance", 3.9, 360),
    ("Pretty Woman", 1990, "Comedy|Drama|Romance", 3.7, 380),
    ("My Best Friend's Wedding", 1997, "Comedy|Drama|Romance", 3.5, 280),
    ("The Mask", 1994, "Action|Comedy|Crime|Fantasy|Thriller", 3.5, 340),
    ("Dumb and Dumber", 1994, "Adventure|Comedy", 3.5, 360),
    ("Home Alone", 1990, "Children|Comedy|Crime", 3.9, 430),
    ("Mrs. Doubtfire", 1993, "Comedy|Drama", 3.8, 390),
    ("The Truman Show", 1998, "Comedy|Drama|Sci-Fi", 4.3, 490),
    ("Groundhog Day", 1993, "Comedy|Drama|Fantasy|Romance", 4.1, 440),
    ("Liar Liar", 1997, "Comedy|Drama|Fantasy", 3.5, 340),
    ("Bruce Almighty", 2003, "Comedy|Drama|Fantasy|Romance", 3.4, 350),
    ("The 40-Year-Old Virgin", 2005, "Comedy|Drama|Romance", 3.5, 300),
    ("Knocked Up", 2007, "Comedy|Drama|Romance", 3.5, 290),
    ("Superbad", 2007, "Comedy", 3.7, 370),
    ("Mean Girls", 2004, "Comedy|Drama", 3.8, 400),
    ("Clueless", 1995, "Comedy|Romance", 3.7, 350),
    ("Ferris Bueller's Day Off", 1986, "Comedy|Drama", 4.0, 420),
    ("Breakfast Club", 1985, "Comedy|Drama", 4.1, 440),
    ("Back to the Future", 1985, "Adventure|Comedy|Sci-Fi", 4.4, 570),
    ("Back to the Future Part II", 1989, "Adventure|Comedy|Sci-Fi", 3.9, 430),
    ("Ghostbusters", 1984, "Action|Comedy|Fantasy|Horror|Sci-Fi", 4.1, 490),
    ("Bill & Ted's Excellent Adventure", 1989, "Adventure|Comedy|Sci-Fi", 3.5, 310),
    ("Wayne's World", 1992, "Comedy|Music", 3.6, 330),
    ("Happy Gilmore", 1996, "Comedy", 3.5, 350),
    ("Billy Madison", 1995, "Comedy", 3.2, 310),
    ("The Wedding Singer", 1998, "Comedy|Drama|Music|Romance", 3.7, 340),
    ("Big Daddy", 1999, "Comedy|Drama|Romance", 3.3, 300),
    ("Shrek", 2001, "Animation|Adventure|Children|Comedy|Fantasy|Romance", 3.9, 500),
    ("Shrek 2", 2004, "Animation|Adventure|Children|Comedy|Fantasy|Romance", 3.8, 470),
    ("Finding Nemo", 2003, "Animation|Children|Comedy", 4.2, 530),
    ("The Lion King", 1994, "Animation|Children|Drama|Musical", 4.3, 570),
    ("Aladdin", 1992, "Animation|Adventure|Children|Comedy|Musical|Romance", 4.0, 500),
    ("Beauty and the Beast", 1991, "Animation|Children|Fantasy|Musical|Romance", 4.1, 510),
    ("The Little Mermaid", 1989, "Animation|Children|Fantasy|Musical|Romance", 3.9, 450),
    ("Mulan", 1998, "Animation|Adventure|Children|Drama|Musical", 3.9, 430),
    ("Pocahontas", 1995, "Animation|Children|Drama|Musical|Romance|War", 3.5, 360),
    ("Tarzan", 1999, "Action|Adventure|Animation|Children|Romance", 3.7, 380),
    ("Hercules", 1997, "Adventure|Animation|Children|Comedy|Musical|Romance", 3.6, 350),
    ("The Jungle Book", 1967, "Animation|Children|Musical", 3.8, 370),
    ("Dumbo", 1941, "Animation|Children|Drama|Musical", 3.8, 310),
    ("Bambi", 1942, "Animation|Children|Drama", 3.8, 290),
    ("Pinocchio", 1940, "Animation|Children|Drama|Fantasy|Musical", 3.8, 280),
    ("Up", 2009, "Adventure|Animation|Children|Drama", 4.4, 540),
    ("WALL-E", 2008, "Adventure|Animation|Children|Romance|Sci-Fi", 4.3, 530),
    ("Ratatouille", 2007, "Animation|Children|Comedy|Drama|Romance", 4.1, 480),
    ("The Incredibles", 2004, "Action|Adventure|Animation|Children|Comedy", 4.2, 520),
    ("Monsters, Inc.", 2001, "Animation|Children|Comedy|Fantasy", 4.1, 500),
    ("Cars", 2006, "Animation|Children|Comedy", 3.6, 420),
    ("Inside Out", 2015, "Adventure|Animation|Children|Comedy|Drama|Fantasy", 4.3, 520),
    ("Coco", 2017, "Adventure|Animation|Children|Comedy|Fantasy|Music|Mystery", 4.4, 490),
    ("Moana", 2016, "Adventure|Animation|Children|Comedy|Fantasy|Musical", 4.2, 460),
    ("Frozen", 2013, "Adventure|Animation|Children|Comedy|Fantasy|Musical|Romance", 4.0, 510),
    ("Tangled", 2010, "Adventure|Animation|Children|Comedy|Fantasy|Musical|Romance", 4.1, 480),
    ("Brave", 2012, "Action|Adventure|Animation|Children|Comedy|Fantasy", 3.7, 400),
    ("Zootopia", 2016, "Action|Adventure|Animation|Children|Comedy|Crime|Mystery", 4.2, 480),
    ("Wreck-It Ralph", 2012, "Adventure|Animation|Children|Comedy|Fantasy", 4.0, 440),
    ("How to Train Your Dragon", 2010, "Adventure|Animation|Children|Fantasy", 4.2, 490),
    ("Kung Fu Panda", 2008, "Action|Adventure|Animation|Children|Comedy", 3.9, 440),
    ("Madagascar", 2005, "Adventure|Animation|Children|Comedy|Fantasy", 3.4, 390),
    ("Ice Age", 2002, "Adventure|Animation|Children|Comedy", 3.6, 400),
    ("Despicable Me", 2010, "Animation|Children|Comedy|Crime", 3.8, 450),
    ("Minions", 2015, "Adventure|Animation|Comedy|Family|Fantasy", 3.4, 400),
    ("The Secret Life of Pets", 2016, "Adventure|Animation|Comedy", 3.5, 370),
    ("Sing", 2016, "Animation|Children|Comedy|Drama|Musical", 3.5, 350),
    ("Alien", 1979, "Horror|Sci-Fi", 4.2, 480),
    ("Aliens", 1986, "Action|Adventure|Horror|Sci-Fi|Thriller", 4.3, 510),
    ("Prometheus", 2012, "Adventure|Sci-Fi|Thriller", 3.6, 390),
    ("Predator", 1987, "Action|Adventure|Sci-Fi|Thriller", 4.0, 420),
    ("Terminator 2: Judgment Day", 1991, "Action|Sci-Fi|Thriller", 4.3, 530),
    ("The Terminator", 1984, "Action|Sci-Fi|Thriller", 4.1, 490),
    ("RoboCop", 1987, "Action|Crime|Drama|Sci-Fi|Thriller", 3.8, 360),
    ("Total Recall", 1990, "Action|Adventure|Sci-Fi|Thriller", 3.7, 370),
    ("Blade Runner", 1982, "Sci-Fi|Thriller", 4.1, 480),
    ("Blade Runner 2049", 2017, "Drama|Mystery|Sci-Fi|Thriller", 4.1, 430),
    ("Star Wars: Episode IV - A New Hope", 1977, "Action|Adventure|Sci-Fi", 4.4, 600),
    ("Star Wars: Episode V - The Empire Strikes Back", 1980, "Action|Adventure|Sci-Fi", 4.5, 590),
    ("Star Wars: Episode VI - Return of the Jedi", 1983, "Action|Adventure|Sci-Fi", 4.2, 550),
    ("Star Wars: Episode I - The Phantom Menace", 1999, "Action|Adventure|Sci-Fi", 3.1, 490),
    ("Star Wars: Episode VII - The Force Awakens", 2015, "Action|Adventure|Sci-Fi|Fantasy", 3.9, 520),
    ("Guardians of the Galaxy", 2014, "Action|Adventure|Comedy|Sci-Fi", 4.2, 510),
    ("Guardians of the Galaxy Vol. 2", 2017, "Action|Adventure|Comedy|Sci-Fi", 3.9, 460),
    ("Doctor Strange", 2016, "Action|Adventure|Fantasy|Sci-Fi", 3.8, 420),
    ("Black Panther", 2018, "Action|Adventure|Sci-Fi", 4.0, 480),
    ("Thor: Ragnarok", 2017, "Action|Adventure|Comedy|Fantasy|Sci-Fi", 4.1, 480),
    ("Ant-Man", 2015, "Action|Adventure|Comedy|Fantasy|Sci-Fi", 3.8, 420),
    ("The Silence of the Lambs", 1991, "Crime|Horror|Thriller", 4.3, 520),
    ("Se7en", 1995, "Crime|Mystery|Thriller", 4.4, 550),
    ("Zodiac", 2007, "Crime|Drama|Mystery|Thriller", 4.0, 390),
    ("Memento", 2000, "Mystery|Thriller", 4.3, 510),
    ("The Prestige", 2006, "Drama|Mystery|Sci-Fi|Thriller", 4.3, 520),
    ("Prisoners", 2013, "Crime|Drama|Mystery|Thriller", 4.2, 460),
    ("Gone Girl", 2014, "Drama|Mystery|Thriller", 4.2, 490),
    ("Shutter Island", 2010, "Drama|Mystery|Thriller", 4.1, 490),
    ("The Sixth Sense", 1999, "Drama|Horror|Mystery", 4.2, 520),
    ("Get Out", 2017, "Horror|Mystery|Thriller", 4.2, 460),
    ("Hereditary", 2018, "Drama|Horror|Mystery|Thriller", 3.9, 360),
    ("A Quiet Place", 2018, "Horror|Sci-Fi|Thriller", 4.0, 430),
    ("It", 2017, "Horror", 3.9, 430),
    ("The Conjuring", 2013, "Horror|Mystery|Thriller", 3.9, 400),
    ("Paranormal Activity", 2007, "Horror|Mystery", 3.1, 300),
    ("The Grudge", 2004, "Horror|Mystery|Thriller", 3.0, 270),
    ("The Ring", 2002, "Horror|Mystery|Thriller", 3.6, 340),
    ("A Nightmare on Elm Street", 1984, "Horror", 3.7, 360),
    ("Halloween", 1978, "Horror|Thriller", 3.9, 380),
    ("Friday the 13th", 1980, "Horror", 3.2, 290),
    ("Scream", 1996, "Horror|Mystery|Thriller", 3.9, 410),
    ("The Exorcist", 1973, "Horror", 4.0, 430),
    ("Jaws", 1975, "Action|Horror|Thriller", 4.1, 450),
    ("Psycho", 1960, "Crime|Horror|Mystery|Thriller", 4.3, 440),
    ("Jurassic Park", 1993, "Action|Adventure|Sci-Fi|Thriller", 4.2, 550),
    ("Jurassic World", 2015, "Action|Adventure|Sci-Fi|Thriller", 3.5, 430),
    ("King Kong", 2005, "Action|Adventure|Drama|Romance", 3.6, 350),
    ("Godzilla", 2014, "Action|Adventure|Sci-Fi|Thriller", 3.2, 330),
    ("Pacific Rim", 2013, "Action|Adventure|Sci-Fi", 3.6, 370),
    ("Transformers", 2007, "Action|Adventure|Sci-Fi|Thriller", 3.2, 390),
    ("Mission: Impossible", 1996, "Action|Adventure|Mystery|Thriller", 3.6, 360),
    ("Mission: Impossible - Fallout", 2018, "Action|Adventure|Thriller", 4.2, 450),
    ("Die Hard", 1988, "Action|Adventure|Thriller", 4.2, 510),
    ("Speed", 1994, "Action|Adventure|Crime|Thriller", 3.7, 370),
    ("Top Gun", 1986, "Action|Drama|Romance", 3.7, 400),
    ("Maverick", 2022, "Action|Drama", 4.4, 480),
    ("John Wick", 2014, "Action|Crime|Thriller", 4.2, 490),
    ("John Wick: Chapter 2", 2017, "Action|Crime|Thriller", 4.1, 450),
    ("The Bourne Identity", 2002, "Action|Adventure|Mystery|Thriller", 4.0, 450),
    ("Casino Royale", 2006, "Action|Adventure|Thriller", 4.2, 500),
    ("Skyfall", 2012, "Action|Adventure|Thriller", 4.1, 480),
    ("Spectre", 2015, "Action|Adventure|Thriller", 3.5, 390),
    ("Dunkirk", 2017, "Action|Drama|History|War", 4.2, 450),
    ("Saving Private Ryan", 1998, "Action|Drama|War", 4.4, 560),
    ("Apocalypse Now", 1979, "Drama|War", 4.2, 440),
    ("Full Metal Jacket", 1987, "Drama|War", 4.1, 430),
    ("Platoon", 1986, "Drama|War", 4.1, 410),
    ("The Hurt Locker", 2008, "Action|Drama|Thriller|War", 4.0, 390),
    ("Zero Dark Thirty", 2012, "Drama|History|Thriller|War", 3.9, 380),
    ("American Sniper", 2014, "Action|Biography|Drama|War", 3.9, 420),
    ("Hacksaw Ridge", 2016, "Biography|Drama|History|War", 4.2, 430),
    ("1917", 2019, "Drama|War", 4.3, 450),
    ("The Pianist", 2002, "Biography|Drama|Music|War", 4.3, 430),
    ("Life is Beautiful", 1997, "Comedy|Drama|Romance|War", 4.4, 460),
    ("Grave of the Fireflies", 1988, "Animation|Drama|War", 4.4, 380),
    ("Spirited Away", 2001, "Adventure|Animation|Family|Fantasy", 4.5, 530),
    ("My Neighbor Totoro", 1988, "Animation|Family|Fantasy", 4.3, 440),
    ("Princess Mononoke", 1997, "Action|Adventure|Animation|Drama|Fantasy", 4.4, 480),
    ("Howl's Moving Castle", 2004, "Adventure|Animation|Drama|Fantasy|Romance|War", 4.4, 450),
    ("Castle in the Sky", 1986, "Action|Adventure|Animation|Fantasy|Romance|Sci-Fi", 4.3, 390),
    ("Nausicaa of the Valley of the Wind", 1984, "Action|Adventure|Animation|Drama|Fantasy|Sci-Fi", 4.3, 380),
    ("Wolf Children", 2012, "Animation|Drama|Fantasy|Romance", 4.3, 340),
    ("Your Name", 2016, "Animation|Drama|Fantasy|Romance", 4.5, 440),
    ("A Silent Voice", 2016, "Animation|Drama|Romance", 4.4, 400),
    ("Akira", 1988, "Action|Adventure|Animation|Sci-Fi", 4.2, 400),
    ("Ghost in the Shell", 1995, "Animation|Action|Sci-Fi", 4.1, 390),
    ("Parasite", 2019, "Comedy|Drama|Thriller", 4.5, 500),
    ("Oldboy", 2003, "Action|Drama|Mystery|Thriller", 4.3, 430),
    ("Train to Busan", 2016, "Action|Horror|Thriller", 4.2, 420),
    ("Crouching Tiger Hidden Dragon", 2000, "Action|Drama|Fantasy|Romance", 4.1, 400),
    ("Bicycle Thieves", 1948, "Drama", 4.4, 360),
    ("Cinema Paradiso", 1988, "Drama|Romance", 4.4, 390),
    ("Amelie", 2001, "Comedy|Romance", 4.3, 490),
    ("Pan's Labyrinth", 2006, "Drama|Fantasy|War", 4.3, 480),
    ("City of God", 2002, "Action|Crime|Drama", 4.4, 470),
    ("The Grand Budapest Hotel", 2014, "Adventure|Comedy|Crime|Drama|Mystery|Romance", 4.2, 470),
    ("Moonrise Kingdom", 2012, "Adventure|Comedy|Drama|Romance", 4.0, 400),
    ("The Royal Tenenbaums", 2001, "Comedy|Drama|Romance", 4.0, 400),
    ("Rushmore", 1998, "Comedy|Drama|Romance", 3.9, 350),
    ("Eternal Sunshine of the Spotless Mind", 2004, "Drama|Romance|Sci-Fi", 4.3, 520),
    ("Lost in Translation", 2003, "Drama|Romance", 3.9, 390),
    ("Her", 2013, "Drama|Romance|Sci-Fi", 4.3, 500),
    ("The Social Network", 2010, "Biography|Drama", 4.3, 500),
    ("Whiplash", 2014, "Drama|Music", 4.4, 500),
    ("Birdman", 2014, "Comedy|Drama", 4.0, 420),
    ("Moonlight", 2016, "Drama", 4.2, 420),
    ("12 Years a Slave", 2013, "Biography|Drama|History", 4.3, 460),
    ("Django Unchained", 2012, "Action|Drama|Western", 4.3, 540),
    ("Inglourious Basterds", 2009, "Adventure|Drama|War", 4.3, 540),
    ("Reservoir Dogs", 1992, "Crime|Mystery|Thriller", 4.3, 500),
    ("Kill Bill: Vol. 1", 2003, "Action|Crime|Thriller", 4.2, 510),
    ("Kill Bill: Vol. 2", 2004, "Action|Crime|Drama|Thriller", 4.1, 470),
    ("Hateful Eight", 2015, "Crime|Drama|Mystery|Western", 4.0, 420),
    ("Once Upon a Time in Hollywood", 2019, "Comedy|Drama", 4.0, 430),
    ("Avatar", 2009, "Action|Adventure|Fantasy|Sci-Fi", 3.7, 590),
    ("Gravity", 2013, "Drama|Sci-Fi|Thriller", 3.9, 480),
    ("The Martian", 2015, "Drama|Sci-Fi", 4.2, 500),
    ("Arrival", 2016, "Drama|Mystery|Sci-Fi", 4.3, 510),
    ("Ex Machina", 2014, "Drama|Sci-Fi|Thriller", 4.2, 460),
    ("Moon", 2009, "Drama|Mystery|Sci-Fi", 4.1, 380),
    ("Annihilation", 2018, "Adventure|Drama|Horror|Mystery|Sci-Fi|Thriller", 3.9, 390),
    ("Ready Player One", 2018, "Action|Adventure|Sci-Fi", 3.8, 430),
    ("Edge of Tomorrow", 2014, "Action|Adventure|Sci-Fi", 4.1, 460),
    ("Elysium", 2013, "Action|Drama|Sci-Fi|Thriller", 3.5, 370),
    ("District 9", 2009, "Action|Drama|Sci-Fi|Thriller", 4.1, 460),
    ("The Hunger Games", 2012, "Action|Adventure|Sci-Fi|Thriller", 3.7, 460),
    ("Divergent", 2014, "Action|Adventure|Sci-Fi|Thriller", 3.2, 360),
    ("Harry Potter and the Sorcerer's Stone", 2001, "Adventure|Children|Fantasy", 4.1, 560),
    ("Harry Potter and the Chamber of Secrets", 2002, "Adventure|Children|Fantasy", 3.9, 510),
    ("Harry Potter and the Prisoner of Azkaban", 2004, "Adventure|Children|Fantasy", 4.3, 560),
    ("Harry Potter and the Goblet of Fire", 2005, "Adventure|Children|Fantasy|Thriller", 4.1, 530),
    ("Harry Potter and the Order of the Phoenix", 2007, "Action|Adventure|Children|Fantasy", 4.0, 510),
    ("Harry Potter and the Deathly Hallows - Part 2", 2011, "Action|Adventure|Children|Fantasy|Mystery", 4.3, 540),
    ("The Lord of the Rings: The Fellowship of the Ring", 2001, "Adventure|Fantasy", 4.5, 640),
    ("The Lord of the Rings: The Two Towers", 2002, "Adventure|Fantasy", 4.5, 620),
    ("The Lord of the Rings: The Return of the King", 2003, "Action|Adventure|Drama|Fantasy", 4.5, 640),
    ("The Hobbit: An Unexpected Journey", 2012, "Adventure|Fantasy", 3.7, 470),
    ("The Chronicles of Narnia: The Lion, the Witch and the Wardrobe", 2005, "Adventure|Children|Drama|Fantasy", 3.7, 390),
    ("Stardust", 2007, "Adventure|Comedy|Fantasy|Romance", 4.0, 380),
    ("The Princess Bride", 1987, "Action|Adventure|Comedy|Fantasy|Romance", 4.4, 490),
    ("Willow", 1988, "Action|Adventure|Fantasy", 3.8, 320),
    ("Labyrinth", 1986, "Adventure|Children|Fantasy|Musical", 3.9, 350),
    ("The NeverEnding Story", 1984, "Adventure|Children|Fantasy", 3.9, 330),
    ("Edward Scissorhands", 1990, "Drama|Fantasy|Romance", 4.1, 430),
    ("Big Fish", 2003, "Adventure|Drama|Fantasy|Romance", 4.2, 430),
    ("Alice in Wonderland", 2010, "Adventure|Family|Fantasy", 3.4, 380),
    ("Charlie and the Chocolate Factory", 2005, "Adventure|Children|Comedy|Fantasy|Musical", 3.5, 400),
    ("Beetlejuice", 1988, "Comedy|Fantasy|Horror", 3.9, 400),
    ("Sleepy Hollow", 1999, "Crime|Fantasy|Horror|Mystery|Thriller", 3.7, 360),
    ("Sweeney Todd: The Demon Barber of Fleet Street", 2007, "Crime|Drama|Fantasy|Horror|Musical|Thriller", 4.0, 370),
    ("Moulin Rouge!", 2001, "Drama|Musical|Romance", 4.0, 430),
    ("Les Misérables", 2012, "Drama|Musical|Romance|War", 3.9, 400),
    ("Chicago", 2002, "Comedy|Crime|Drama|Musical", 3.9, 390),
    ("Mamma Mia!", 2008, "Comedy|Drama|Musical|Romance", 3.4, 340),
    ("Grease", 1978, "Comedy|Drama|Musical|Romance", 3.8, 410),
    ("West Side Story", 1961, "Crime|Drama|Musical|Romance", 4.1, 360),
    ("Singin' in the Rain", 1952, "Comedy|Musical|Romance", 4.3, 380),
    ("The Sound of Music", 1965, "Drama|Musical|Romance|War", 4.2, 390),
    ("Bohemian Rhapsody", 2018, "Biography|Drama|Music", 4.0, 450),
    ("Rocketman", 2019, "Biography|Drama|Fantasy|Music|Musical|Romance", 4.0, 400),
    ("Walk the Line", 2005, "Biography|Drama|Music|Romance", 4.1, 400),
    ("Ray", 2004, "Biography|Drama|Music", 4.1, 390),
    ("Almost Famous", 2000, "Adventure|Drama|Music|Romance", 4.2, 410),
    ("School of Rock", 2003, "Children|Comedy|Music", 3.9, 400),
    ("Whiplash", 2014, "Drama|Music", 4.4, 500),
    ("Black Swan", 2010, "Drama|Horror|Thriller", 4.1, 490),
    ("Natalie Portman - Closer", 2004, "Drama|Romance", 3.9, 360),
    ("Requiem for a Dream", 2000, "Drama|Thriller", 4.2, 480),
    ("Trainspotting", 1996, "Crime|Drama", 4.3, 490),
    ("American History X", 1998, "Crime|Drama", 4.4, 510),
    ("A Beautiful Mind", 2001, "Biography|Drama|Mystery|Romance|Thriller", 4.2, 480),
    ("The Theory of Everything", 2014, "Biography|Drama|Romance", 4.1, 430),
    ("The Imitation Game", 2014, "Biography|Drama|Thriller|War", 4.3, 510),
    ("Hidden Figures", 2016, "Biography|Drama|History", 4.3, 460),
    ("The Wolf of Wall Street", 2013, "Biography|Comedy|Crime|Drama", 4.2, 520),
    ("Catch Me If You Can", 2002, "Biography|Crime|Drama", 4.1, 470),
    ("The Big Short", 2015, "Biography|Comedy|Drama", 4.2, 450),
    ("Moneyball", 2011, "Biography|Drama|Sport", 4.0, 430),
    ("Remember the Titans", 2000, "Biography|Drama|Sport", 4.1, 430),
    ("Rocky", 1976, "Drama|Sport", 4.2, 460),
    ("Rocky II", 1979, "Drama|Sport", 3.9, 360),
    ("The Karate Kid", 1984, "Drama|Sport", 3.9, 410),
    ("Whiplash", 2014, "Drama|Music", 4.4, 500),
]


# LOAD DATA
def load_data():
    """Load the movie dataset."""
    records = []
    for i, (title, year, genres, avg_rating, num_ratings) in enumerate(MOVIES_DATA):
        records.append({
            "movieId": i + 1,
            "title": f"{title} ({year})",
            "genres": genres,
            "avg_rating": avg_rating,
            "num_ratings": num_ratings,
            "year": float(year)
        })
    movies = pd.DataFrame(records)
    ratings = pd.DataFrame({"userId": [], "movieId": [], "rating": []})
    return movies, ratings


# FEATURE ENGINEERING
def build_features(movies: pd.DataFrame, ratings: pd.DataFrame):
    """Create features for clustering. Uses only genres for clustering."""
    movies = movies.copy()
    movies["genres_list"] = movies["genres"].str.split("|")
    movies = movies[movies["genres"] != "(no genres listed)"]

    # Genre encoding
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(movies["genres_list"]),
        columns=mlb.classes_,
        index=movies.index
    )

    # Numeric feature
    scaler = StandardScaler()
    numeric = scaler.fit_transform(movies[["avg_rating", "num_ratings", "year"]])
    numeric_df = pd.DataFrame(numeric, columns=["avg_rating", "num_ratings", "year"], index=movies.index)

    features = genre_matrix   # Only genres used for clustering
    return movies.reset_index(drop=True), features, mlb.classes_


# CLUSTERING
def find_optimal_k(features: pd.DataFrame, k_range=range(4, 12)):
    """Find best number of clusters using Elbow and Silhouette methods."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        sil = silhouette_score(features, labels, sample_size=min(300, len(features)))
        silhouettes.append(sil)
    return list(k_range), inertias, silhouettes
    
def cluster_movies(features: pd.DataFrame, n_clusters: int = 18):
    """Run KMeans clustering on genre features."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=40, max_iter=500)
    labels = km.fit_predict(features)
    return km, labels


def reduce_dimensions(features: pd.DataFrame):
    """Reduce features to 2D using PCA for visualization."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)
    return coords, pca.explained_variance_ratio_


def auto_name_clusters(movies_df, labels, n_clusters=18):
    """Automatically name clusters based on dominant genres."""
    GENRE_EMOJIS = {
        "Action": "💥", "Comedy": "😂", "Drama": "💔",
        "Sci-Fi": "🚀", "Thriller": "😱", "Horror": "👻",
        "Romance": "💑", "Animation": "🎨", "Adventure": "🗺️",
        "Crime": "🔫", "Fantasy": "🧙", "War": "⚔️",
        "Musical": "🎵", "Mystery": "🔍", "Biography": "📖",
        "Children": "🧸", "Western": "🤠", "Sport": "🏆",
        "History": "📜", "Music": "🎸", "Documentary": "🎥",
    }

    cluster_names = {}
    cluster_descriptions = {}

    for cluster_id in range(n_clusters):
        cluster_movies = movies_df[labels == cluster_id]
        genre_counts = {}
        for genres in cluster_movies["genres_list"]:
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if not genre_counts:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            cluster_descriptions[cluster_id] = "Mixed collection of films."
            continue

        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        top = sorted_genres[0][0]
        second = sorted_genres[1][0] if len(sorted_genres) > 1 else ""
        emoji = GENRE_EMOJIS.get(top, "🎬")

        cluster_names[cluster_id] = f"{emoji} {top} & {second}" if second else f"{emoji} {top}"
        cluster_descriptions[cluster_id] = (
            f"Dominated by {top} ({genre_counts[top]} movies) · "
            f"Avg rating: {cluster_movies['avg_rating'].mean():.2f}⭐"
        )

    return cluster_names, cluster_descriptions


# RECOMMENDATION
def recommend(movie_title: str, movies_df: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray, top_n: int = 8):
    """Recommend similar movies from the same cluster."""
    mask = movies_df["title"].str.lower().str.contains(movie_title.lower(), na=False)
    matches = movies_df[mask]

    if matches.empty:
        return None, None, []

    idx = matches.index[0]
    movie = movies_df.loc[idx]
    cluster_id = labels[idx]

    cluster_mask = labels == cluster_id
    cluster_movies_df = movies_df[cluster_mask].copy()
    cluster_movies_df = cluster_movies_df[cluster_movies_df.index != idx]

    cluster_movies_df["score"] = (
        cluster_movies_df["avg_rating"] * np.log1p(cluster_movies_df["num_ratings"])
    )
    recs = cluster_movies_df.nlargest(top_n, "score")
    return movie, cluster_id, recs


# MAIN PIPELINE
def run_pipeline():
    """Run the full pipeline: load data, create features, cluster, and prepare results."""
    movies_raw, ratings_raw = load_data()
    movies_df, features, genre_cols = build_features(movies_raw, ratings_raw)

    km_model, labels = cluster_movies(features, n_clusters=18)
    coords, var_ratio = reduce_dimensions(features)

    movies_df["cluster"] = labels
    movies_df["pca_x"] = coords[:, 0]
    movies_df["pca_y"] = coords[:, 1]

    global CLUSTER_NAMES, CLUSTER_DESCRIPTIONS
    CLUSTER_NAMES, CLUSTER_DESCRIPTIONS = auto_name_clusters(movies_df, labels, n_clusters=18)

    return movies_df, features, km_model, labels, coords, var_ratio, genre_cols
