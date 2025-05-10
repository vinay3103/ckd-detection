# myapp/management/commands/train_model.py
from django.core.management.base import BaseCommand
from ckd_app.model import train_model  # Adjust the import to match your project structure

class Command(BaseCommand):
    help = 'Train the machine learning model'

    def handle(self, *args, **options):
        self.stdout.write('Training the model...')
        voting_clf, scaler, label_encoder = train_model()
        self.stdout.write(self.style.SUCCESS('Model trained successfully!'))
