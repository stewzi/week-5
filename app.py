import streamlit as st

from apputil import (
    survival_demographics, visualize_demographic, family_groups,
    last_names, visualize_families, determine_age_division,
    visualize_age_division,
)


st.write('# Titanic Visualization 1')
st.dataframe(survival_demographics(), hide_index=True)
st.write('How do survival rates differ between women and men across age groups and passenger classes?')
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)
st.write('Passengers with missing ages are excluded from the age groups. Empty groups have an undefined survival rate.')

st.write('# Titanic Visualization 2')
st.dataframe(family_groups(), hide_index=True)
st.write('How does average ticket fare change with family size in each passenger class?')
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)
st.write('Passenger counts and minimum and maximum fares are available when hovering over the chart.')

surname_counts = last_names()
st.write('### Passenger counts by last name')
st.dataframe(surname_counts)
st.write(
    'Surname counts do not exactly match family sizes. Family size includes siblings, '
    'spouses, parents, and children, who may have different last names. Unrelated '
    'passengers can also share a surname, and some relatives may not appear in this '
    'dataset. For example, the Andersson surname occurs '
    f'{surname_counts.get("Andersson", 0)} times, so a surname alone is not a reliable family identifier.'
)

st.write('# Titanic Visualization Bonus')
st.dataframe(determine_age_division(), hide_index=True)
st.write('Within each passenger class, did passengers above the class median age have a lower survival rate?')
fig3 = visualize_age_division()
st.plotly_chart(fig3, use_container_width=True)
st.write('Missing ages remain missing in older_passenger and are excluded from this chart. Each class has its own median age.')
