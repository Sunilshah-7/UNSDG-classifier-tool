import React from "react";

const NoSdgPage: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br">
      <div className="text-center space-y-6 max-w-xl px-6">
        <h1 className="text-3xl sm:text-4xl font-bold text-black">
          This project does not satisfy any SDG
        </h1>
        <p className="text-gray-600 text-lg">
          We couldn’t find SDG matches above the relevance threshold for the
          provided repository/description.
        </p>
      </div>
    </div>
  );
};

export default NoSdgPage;

