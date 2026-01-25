import type { Meta, StoryObj } from '@storybook/react';
import { MoreButton } from '../components/more-button';

const meta = {
  title: 'Components/MoreButton',
  component: MoreButton,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    'aria-label': { control: 'text' },
  },
} satisfies Meta<typeof MoreButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const WithCustomLabel: Story = {
  args: {
    'aria-label': 'Custom Action',
  },
};
