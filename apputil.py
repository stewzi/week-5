import plotly.express as px
import pandas as pd


DATA_URL = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'
AGE_GROUPS = ['Child', 'Teen', 'Adult', 'Senior']


def load_data():
    """Load Titanic data and normalize column names."""
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    return df


def survival_demographics():
    """Return survival statistics for every class, sex, and age group."""
    df = load_data()
    # Use whole-year ages: children are under 13, teens under 20, etc.
    # Missing ages remain missing instead of being assigned an age group.
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 13, 20, 60, float('inf')],
        labels=AGE_GROUPS, right=False
    )
    results = df.groupby(['pclass', 'sex', 'age_group'], observed=False).agg(
        n_passengers=('survived', 'size'),
        n_survivors=('survived', 'sum')
    ).reset_index()
    results['survival_rate'] = results['n_survivors'] / results['n_passengers']
    return results.sort_values(['pclass', 'sex', 'age_group']).reset_index(drop=True)


def visualize_demographic():
    """Compare survival by sex and age within each passenger class."""
    results = survival_demographics()
    fig = px.bar(
        results, x='age_group', y='survival_rate', color='sex',
        facet_col='pclass', barmode='group',
        category_orders={'age_group': AGE_GROUPS, 'sex': ['female', 'male']},
        hover_data=['n_passengers', 'n_survivors'],
        labels={'age_group': 'Age group', 'survival_rate': 'Survival rate',
                'sex': 'Sex', 'pclass': 'Passenger class'},
        title='Survival rates by age, sex, and passenger class'
    )
    fig.update_yaxes(tickformat='.0%', range=[0, 1])
    fig.update_traces(hovertemplate=None)
    return fig


def family_groups():
    """Summarize ticket fares by family size and passenger class."""
    df = load_data()
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    results = df.groupby(['family_size', 'pclass']).agg(
        n_passengers=('fare', 'size'),
        avg_fare=('fare', 'mean'),
        min_fare=('fare', 'min'),
        max_fare=('fare', 'max')
    ).reset_index()
    return results.sort_values(['pclass', 'family_size']).reset_index(drop=True)


def last_names():
    """Return the passenger count for each surname."""
    df = load_data()
    surnames = df['name'].str.split(',', n=1).str[0].str.strip()
    counts = surnames.value_counts()
    counts.index.name = 'last_name'
    counts.name = 'n_passengers'
    return counts


def visualize_families():
    """Compare average fares for family sizes across passenger classes."""
    results = family_groups()
    results['pclass'] = results['pclass'].astype(str)
    fig = px.line(
        results, x='family_size', y='avg_fare', color='pclass', markers=True,
        category_orders={'pclass': ['1', '2', '3']},
        hover_data=['n_passengers', 'min_fare', 'max_fare'],
        labels={'family_size': 'Family size (including passenger)',
                'avg_fare': 'Average ticket fare', 'pclass': 'Passenger class'},
        title='Average ticket fare by family size and passenger class'
    )
    fig.update_xaxes(dtick=1)
    return fig


def determine_age_division():
    """Flag ages above the median for the passenger's class."""
    df = load_data()
    class_median = df.groupby('pclass')['age'].transform('median')
    df['older_passenger'] = df['age'] > class_median
    return df


def visualize_age_division():
    """Compare survival above and at or below each class's median age."""
    df = determine_age_division()
    # Unknown ages compare False, but are excluded from the age comparison.
    df = df.dropna(subset=['age']).copy()
    df['age_division'] = df['older_passenger'].map(
        {False: 'At or below class median', True: 'Above class median'}
    )
    results = df.groupby(['pclass', 'age_division']).agg(
        survival_rate=('survived', 'mean'),
        n_passengers=('survived', 'size')
    ).reset_index()
    results['pclass'] = results['pclass'].astype(str)
    fig = px.bar(
        results, x='pclass', y='survival_rate', color='age_division',
        barmode='group', hover_data=['n_passengers'],
        category_orders={'age_division': ['At or below class median', 'Above class median']},
        labels={'pclass': 'Passenger class', 'survival_rate': 'Survival rate',
                'age_division': 'Age compared with class median'},
        title='Survival by age relative to each class median'
    )
    fig.update_yaxes(tickformat='.0%', range=[0, 1])
    return fig
