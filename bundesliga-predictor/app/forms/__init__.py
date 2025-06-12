from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, SubmitField, FormField, FieldList, Form
from wtforms.validators import DataRequired, NumberRange, ValidationError
from ..utils import MAX_POINTS, TEAMS

# Unterformular für Punkte je Team
class TableEntryForm(Form):
    points = IntegerField(
        "Punkte",
        validators=[DataRequired(), NumberRange(min=0, max=MAX_POINTS)],
        render_kw={"class": "form-control", "placeholder": "0"},
    )

# Hauptformular
class PredictionForm(FlaskForm):
    spieltag = IntegerField(
        "Spieltag (1 – 34)",
        validators=[DataRequired(), NumberRange(min=1, max=34)],
        render_kw={"class": "form-control", "placeholder": "z. B. 5"},
    )
    home_team = SelectField(
        "Heimteam",
        choices=[(t, t) for t in sorted(TEAMS)],
        validators=[DataRequired()],
        render_kw={"class": "form-select"},
    )
    away_team = SelectField(
        "Auswärtsteam",
        choices=[(t, t) for t in sorted(TEAMS)],
        validators=[DataRequired()],
        render_kw={"class": "form-select"},
    )
    table = FieldList(
        FormField(TableEntryForm),
        min_entries=len(TEAMS),
        max_entries=len(TEAMS),
    )
    submit = SubmitField(
        "Vorhersage berechnen",
        render_kw={"class": "btn btn-primary btn-lg w-100 mt-3"},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for idx, entry in enumerate(self.table):
            entry.label = TEAMS[idx]

    def validate_away_team(self, field):
        if field.data == self.home_team.data:
            raise ValidationError("Heim- und Auswärtsteam müssen verschieden sein.")