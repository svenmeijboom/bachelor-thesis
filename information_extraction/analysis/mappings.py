import json
from pathlib import Path


def reverse_mapping(mapping: dict):
    def reverse(sub_mapping: dict):
        return {
            open_label: closed_label
            for closed_label, open_labels in sub_mapping.items()
            if open_labels is not None
            for open_label in open_labels
        }

    return {
        vertical: {
            website: reverse(website_mapping)
            for website, website_mapping in vertical_mapping.items()
        }
        for vertical, vertical_mapping in mapping.items()
    }


CLOSED_TO_OPEN_MAPPING = {
    'movie': {
        'allmovie': {
            'director': ['Director'],
            'genre': ['Genres'],
            'mpaa_rating': ['MPAA Rating'],
            'title': None,
        },
        'amctv': {
            'director': ['Director:'],
            'genre': ['Genre/Type:'],
            'mpaa_rating': ['MPAA Rating:'],
            'title': None,
        },
        'hollywood': {
            'director': ['Director'],
            'genre': None,
            'mpaa_rating': None,
            'title': None,
        },
        'iheartmovies': {
            'director': ['Directed by'],
            'genre': ['Genres'],
            'mpaa_rating': ['MPAA Rating'],
            'title': None,
        },
        'imdb': {
            'director': ['Director:', 'Directors:'],
            'genre': ['Genres:'],
            'mpaa_rating': ['Motion Picture Rating'],
            'title': None,
        },
        'metacritic': {
            'director': ['Director:'],
            'genre': ['Genre(s):'],
            'mpaa_rating': ['Rating:'],
            'title': None,
        },
        'rottentomatoes': {
            'director': ['Directed By:'],
            'genre': ['Genre:'],
            'mpaa_rating': ['Rated:'],
            'title': None,
        },
        'yahoo': {
            'director': ['Directed by:'],
            'genre': ['Genres:'],
            'mpaa_rating': ['MPAA Rating:'],
            'title': None,
        },
    },
    'nbaplayer': {
        'fanhouse': {
            'height': ['Height'],
            'name': None,
            'team': ['Team'],
            'weight': ['Weight'],
        },
        'foxsports': {
            'height': ['Ht'],
            'name': None,
            'team': None,
            'weight': ['Wt'],
        },
        'espn': {
            'height': ['Height'],
            'name': None,
            'team': None,
            'weight': ['Weight'],
        },
        'msnca': {
            'height': ['Height:'],
            'name': None,
            'team': ['Team:'],
            'weight': ['Weight:'],
        },
        'si': {
            'height': ['Height:'],
            'name': None,
            'team': None,
            'weight': ['Weight:'],
        },
        'slam': {
            'height': ['Height'],
            'name': None,
            'team': None,
            'weight': ['Weight'],
        },
        'usatoday': {
            'height': ['Height:'],
            'name': None,
            'team': None,
            'weight': ['Weight:'],
        },
        'yahoo': {
            'height': ['Height'],
            'name': None,
            'team': None,
            'weight': ['Weight'],
        }
    },
    'university': {
        'collegeprowler': {
            'name': None,
            'phone': None,
            'website': None,
            'type': ['Control:'],
        },
        'ecampustours': {
            'name': None,
            'phone': None,
            'website': None,
            'type': ['Affiliation'],
        },
        'embark': {
            'name': None,
            'phone': ['Phone:'],
            'website': None,
            'type': ['School Type:'],
        },
        'matchcollege': {
            'name': None,
            'phone': ['General Phone:'],
            'website': ['Website'],
            'type': ['Type:'],
        },
        'usnews': {
            'name': None,
            'phone': None,
            'website': ['Web site:'],
            'type': ['Institutional Control:'],
        },
    },
}

OPEN_TO_CLOSED_MAPPING = reverse_mapping(CLOSED_TO_OPEN_MAPPING)

with open(Path(__file__).parent / 'webke_split_mapping.json') as _file:
    WEBKE_SPLIT_MAPPING = json.load(_file)
