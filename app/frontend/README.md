# Age Predictor Frontend

This is a Next.js frontend application for the Age Predictor API. It provides a user-friendly interface to input health metrics and receive age predictions.

## Features

- Step-by-step form wizard asking one question at a time
- Input validation for all fields
- Progress indicator showing completion status
- Smooth transitions between questions
- Loading spinner while waiting for prediction
- Clean result display with option to start over

## Getting Started

### Prerequisites

- Node.js 14.x or later
- npm or yarn
- The Age Predictor API running on http://localhost:8000

### Installation

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
# or
yarn install
```

3. Start the development server:

```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Usage

1. The application will present health-related questions one by one
2. Fill in your information or select from available options
3. Navigate between questions using the Previous and Next buttons
4. After completing all questions, click "Predict Age"
5. View your predicted age based on the provided health metrics
6. Start over if you wish to make changes

## API Connection

The frontend connects to the Age Predictor API at `http://localhost:8000`. Make sure the API is running before using the frontend application.

## Built With

- [Next.js](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Axios](https://axios-http.com/) - HTTP client
