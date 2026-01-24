import type { Meta, StoryObj } from '@storybook/react';
import { I18nextProvider } from 'react-i18next';
import { MoreButton } from '../components/more-button';
import i18n from '../locales/config';
import '../../tailwind.css';

const meta = {
  title: 'Components/MoreButton',
  component: MoreButton,
  decorators: [
    (Story) => (
      <I18nextProvider i18n={i18n}>
        <div className="p-10 bg-gray-100 flex gap-4">
          <div className="group relative w-20 h-20 border border-gray-300 flex items-center justify-center bg-white rounded-md">
            <span>Hover me</span>
            <div className="absolute top-1 right-1">
                <Story />
            </div>
          </div>

          <div className="group relative w-20 h-20 border border-gray-300 flex items-center justify-center bg-white rounded-md">
             <span>Focus me</span>
             <div className="absolute top-1 right-1">
                <Story />
             </div>
          </div>
        </div>
      </I18nextProvider>
    ),
  ],
} satisfies Meta<typeof MoreButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
